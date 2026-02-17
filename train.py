"""
Entraînement YOLO26-seg pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle

Structure identique à Mask R-CNN et DeepLabV3+ pour comparaison équitable
"""

import os
import json
import yaml
import shutil
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
import gc
import torch
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN et DeepLabV3+)
# =============================================================================

CONFIG = {
    # Chemins (à adapter)
    "images_dir": "../dataset1/images/default",
    "annotations_file": "../dataset1/annotations/instances_default.json",
    "output_dir": "./output",
    
    # Classes (dans l'ordre de CVAT) - IDENTIQUE aux autres modèles
    "classes": [
        "toiture_tole_ondulee",  # 0
        "toiture_tole_bac",      # 1
        "toiture_tuile",         # 2
        "toiture_dalle"          # 3
    ],
    
    # Modèle YOLO26-seg
    "model_size": "n",  # n, s, m, l, x
    
    # Hyperparamètres - IDENTIQUES aux autres modèles
    "num_epochs": 25,
    "batch_size": 2,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    
    # Images
    "image_size": 640,
    
    # Dataset - IDENTIQUE aux autres modèles
    "train_split": 0.85,
    
    # Sauvegarde
    "save_every": 5,
}


# =============================================================================
# UTILITAIRES TEMPS (identique aux autres modèles)
# =============================================================================

def format_time(seconds):
    """Formater les secondes en format lisible HH:MM:SS"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


# =============================================================================
# CONVERSION COCO -> YOLO FORMAT
# =============================================================================

def coco_to_yolo_segmentation(coco_annotation, img_width, img_height):
    """
    Convertir une annotation COCO en format YOLO segmentation
    YOLO format: class_id x1 y1 x2 y2 ... xn yn (coordonnées normalisées)
    """
    segmentation = coco_annotation.get('segmentation', [])
    
    if not segmentation or not isinstance(segmentation, list):
        return None
    
    # Prendre le premier polygone s'il y en a plusieurs
    polygon = segmentation[0] if segmentation else []
    
    if len(polygon) < 6:  # Minimum 3 points (6 coordonnées)
        return None
    
    # Normaliser les coordonnées
    normalized_polygon = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        # Clipper entre 0 et 1
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        normalized_polygon.extend([x, y])
    
    return normalized_polygon


def prepare_yolo_dataset(images_dir, annotations_file, output_dir, classes, train_split=0.85):
    """
    Convertir le dataset COCO en format YOLO pour segmentation
    """
    
    print("📂 Préparation du dataset YOLO...")
    
    # Créer la structure de dossiers YOLO
    dataset_dir = os.path.join(output_dir, "dataset")
    train_images_dir = os.path.join(dataset_dir, "images", "train")
    val_images_dir = os.path.join(dataset_dir, "images", "val")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    val_labels_dir = os.path.join(dataset_dir, "labels", "val")
    
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Charger les annotations COCO
    coco = COCO(annotations_file)
    
    # Mapping des catégories COCO vers indices YOLO
    cat_ids = coco.getCatIds()
    cat_mapping = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    
    # Liste des images
    image_ids = list(coco.imgs.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)
    
    # Split train/val
    split_idx = int(len(image_ids) * train_split)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    print(f"   Train: {len(train_ids)} images")
    print(f"   Val: {len(val_ids)} images")
    
    stats = {'train': 0, 'val': 0, 'annotations': 0}
    
    # Traiter les images
    for split_name, img_ids, img_dir, lbl_dir in [
        ('train', train_ids, train_images_dir, train_labels_dir),
        ('val', val_ids, val_images_dir, val_labels_dir)
    ]:
        for img_id in img_ids:
            img_info = coco.imgs[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copier l'image
            src_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(src_path):
                continue
            
            dst_path = os.path.join(img_dir, img_filename)
            shutil.copy2(src_path, dst_path)
            
            # Créer le fichier de labels
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(lbl_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in anns:
                    if ann.get('iscrowd', 0):
                        continue
                    
                    class_id = cat_mapping.get(ann['category_id'])
                    if class_id is None:
                        continue
                    
                    # Convertir la segmentation
                    polygon = coco_to_yolo_segmentation(ann, img_width, img_height)
                    if polygon is None:
                        continue
                    
                    # Écrire: class_id x1 y1 x2 y2 ...
                    polygon_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                    f.write(f"{class_id} {polygon_str}\n")
                    stats['annotations'] += 1
            
            stats[split_name] += 1
    
    # Créer le fichier dataset.yaml
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"   Annotations converties: {stats['annotations']}")
    print(f"   Dataset YOLO créé: {dataset_dir}")
    
    return yaml_path, stats


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_yolo26_seg():
    """Entraîner YOLO26-seg"""
    
    print("=" * 70)
    print("   YOLO26-seg - Segmentation des Toitures Cadastrales")
    print("=" * 70)
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Préparer le dataset
    yaml_path, dataset_stats = prepare_yolo_dataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        CONFIG["output_dir"],
        CONFIG["classes"],
        CONFIG["train_split"]
    )
    
    # Charger le modèle YOLO26-seg pré-entraîné
    model_name = f"yolo26{CONFIG['model_size']}-seg.pt"
    print(f"\n🧠 Chargement du modèle {model_name}...")
    
    gc.collect()

    model = YOLO(model_name)
    
    print(f"   Classes: {CONFIG['classes']}")
    
    # Entraînement
    print("\n" + "=" * 70)
    print("   🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"   📅 Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📊 Epochs: {CONFIG['num_epochs']} | Batch size: {CONFIG['batch_size']}")
    print("=" * 70)
    
    start_time = time.time()
    training_start = datetime.now()
    
    # Lancer l'entraînement
    results = model.train(
        data=yaml_path,
        epochs=CONFIG["num_epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["image_size"],
        lr0=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
        project=CONFIG["output_dir"],
        name="train",
        exist_ok=True,
        seed=42,
        verbose=True,
        save=True,
        save_period=CONFIG["save_every"],
        plots=True,
        # ↓↓↓ AJOUTS POUR CORRIGER LA MÉMOIRE ↓↓↓
        cache=False,   # Ne pas mettre les images en RAM entre les epochs
        workers=0,     # Pas de sous-processus DataLoader (chacun duplique la RAM)
        amp=False,     # Désactiver mixed precision si GPU < 4 Go
    )

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    total_time = time.time() - start_time
    training_end = datetime.now()
    
    # Récupérer les métriques d'entraînement
    train_dir = os.path.join(CONFIG["output_dir"], "train")
    
    # Charger les résultats
    results_csv = os.path.join(train_dir, "results.csv")
    history = {
        'train_loss': [],
        'val_loss': [],
        'mAP50': [],
        'mAP50_95': [],
        'epoch_times': [],
    }
    
    if os.path.exists(results_csv):
        import csv
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Les noms des colonnes peuvent varier selon la version
                try:
                    history['train_loss'].append(float(row.get('train/seg_loss', 0) or 0))
                    history['val_loss'].append(float(row.get('val/seg_loss', 0) or 0))
                    history['mAP50'].append(float(row.get('metrics/mAP50(M)', 0) or 0))
                    history['mAP50_95'].append(float(row.get('metrics/mAP50-95(M)', 0) or 0))
                except:
                    pass
    
    # Calculer le temps moyen par epoch
    avg_epoch_time = total_time / CONFIG["num_epochs"]
    history['epoch_times'] = [avg_epoch_time] * CONFIG["num_epochs"]
    
    # Stats de temps
    time_stats = {
        'total_time': total_time,
        'total_time_formatted': format_time(total_time),
        'avg_epoch_time': avg_epoch_time,
        'avg_epoch_time_formatted': format_time(avg_epoch_time),
        'start_datetime': training_start.strftime("%Y-%m-%d %H:%M:%S"),
        'end_datetime': training_end.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    history['time_stats'] = time_stats
    
    # Sauvegarder l'historique
    history_path = os.path.join(CONFIG["output_dir"], "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Copier le meilleur modèle
    best_model_src = os.path.join(train_dir, "weights", "best.pt")
    best_model_dst = os.path.join(CONFIG["output_dir"], "best_model.pt")
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, best_model_dst)
    
    final_model_src = os.path.join(train_dir, "weights", "last.pt")
    final_model_dst = os.path.join(CONFIG["output_dir"], "final_model.pt")
    if os.path.exists(final_model_src):
        shutil.copy2(final_model_src, final_model_dst)
    
    # Rapport final (identique aux autres modèles)
    print("\n" + "=" * 70)
    print("   🎉 ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    
    best_mAP50 = max(history['mAP50']) if history['mAP50'] else 0
    best_mAP50_95 = max(history['mAP50_95']) if history['mAP50_95'] else 0
    
    print(f"\n📊 RÉSUMÉ DES PERFORMANCES")
    print(f"   {'─' * 50}")
    print(f"   Meilleur mAP@50:    {best_mAP50:.4f}")
    print(f"   Meilleur mAP@50:95: {best_mAP50_95:.4f}")
    
    print(f"\n⏱️  RAPPORT DE TEMPS")
    print(f"   {'─' * 50}")
    print(f"   Début:              {time_stats['start_datetime']}")
    print(f"   Fin:                {time_stats['end_datetime']}")
    print(f"   {'─' * 50}")
    print(f"   ⏱️  Temps total:       {time_stats['total_time_formatted']}")
    print(f"   ⏱️  Temps moyen/epoch: {time_stats['avg_epoch_time_formatted']}")
    
    print(f"\n💾 FICHIERS SAUVEGARDÉS")
    print(f"   {'─' * 50}")
    print(f"   📁 Dossier: {CONFIG['output_dir']}")
    print(f"   ├── best_model.pt")
    print(f"   ├── final_model.pt")
    print(f"   ├── history.json")
    print(f"   └── train/")
    print("=" * 70)
    
    # Sauvegarder le rapport
    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ENTRAÎNEMENT - YOLO26-seg CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        for key, value in CONFIG.items():
            f.write(f"   {key}: {value}\n")
        
        f.write("\nPERFORMANCES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Meilleur mAP@50:    {best_mAP50:.4f}\n")
        f.write(f"   Meilleur mAP@50:95: {best_mAP50_95:.4f}\n")
        
        f.write("\nTEMPS D'ENTRAÎNEMENT\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Début:               {time_stats['start_datetime']}\n")
        f.write(f"   Fin:                 {time_stats['end_datetime']}\n")
        f.write(f"   Temps total:         {time_stats['total_time_formatted']}\n")
        f.write(f"   Temps moyen/epoch:   {time_stats['avg_epoch_time_formatted']}\n")
    
    print(f"\n📄 Rapport sauvegardé: {report_path}")
    
    return model, history


if __name__ == "__main__":
    train_yolo26_seg()
