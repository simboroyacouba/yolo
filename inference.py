"""
Inférence YOLO26-seg - Prédiction sur nouvelles images
Segmentation des toitures cadastrales

Fonctionnalités:
- Temps d'inférence par image
- Résumé global pour les dossiers
- Export des masques
- Rapports JSON détaillés
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import time
import json
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['classes']
    
    # Palette de couleurs auto-générée pour toutes les classes
_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 20, 147), (0, 128, 0),
]
CLASSES = load_classes()

COLORS = {
    cls: _PALETTE[i % len(_PALETTE)]
    for i, cls in enumerate(CLASSES[1:])  # on ignore __background__
}


CONFIG = {
    "model_path": os.getenv("SEGMENTATION_MODEL_PATH", "./runs/segment/train/weights/best.pt"),
    "input_dir": os.getenv("SEGMENTATION_TEST_IMAGES_DIR", "./test_images"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir": os.getenv("SEGMENTATION_OUTPUT_DIR", "./predictions"),
    "score_threshold": 0.5,
    "export_masks": False,
    "show_display": False,
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{int(seconds//60)}m {seconds%60:.1f}s"


# =============================================================================
# MODÈLE
# =============================================================================

def load_model(checkpoint_path):
    model = YOLO(checkpoint_path)
    print(f"✅ Modèle chargé: {checkpoint_path}")
    return model


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    
    start_time = time.time()
    results = model.predict(image_path, conf=threshold, verbose=False)
    inference_time = time.time() - start_time
    
    result = results[0]
    
    predictions = {
        'boxes': [],
        'labels': [],
        'scores': [],
        'masks': [],
        'class_names': [],
        'inference_time': inference_time
    }
    
    if result.masks is not None and len(result.boxes) > 0:
        predictions['boxes'] = result.boxes.xyxy.cpu().numpy()
        predictions['labels'] = result.boxes.cls.cpu().numpy().astype(int)
        predictions['scores'] = result.boxes.conf.cpu().numpy()
        
        masks_data = result.masks.data.cpu().numpy()
        orig_shape = result.orig_shape
        
        for i in range(len(masks_data)):
            mask = masks_data[i]
            mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (orig_shape[1], orig_shape[0]), Image.NEAREST))
            predictions['masks'].append(mask_resized > 0.5)
        
        predictions['class_names'] = [CLASSES[int(l) + 1] for l in predictions['labels']]
    
    return image, predictions


def calculate_surface(mask):
    return int(np.sum(mask > 0))


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize_predictions(image, predictions, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    
    boxes = predictions['boxes']
    scores = predictions['scores']
    masks = predictions['masks']
    class_names = predictions['class_names']
    inference_time = predictions.get('inference_time', 0)
    
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for i in range(len(boxes)):
        class_name = class_names[i]
        color = COLORS.get(class_name, (128, 128, 128))
        color_norm = [c/255 for c in color]
        
        if i < len(masks):
            overlay[masks[i] > 0] = [*color_norm, 0.5]
        
        x1, y1, x2, y2 = boxes[i]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                  edgecolor=color_norm, facecolor='none')
        axes[1].add_patch(rect)
        
        surface = calculate_surface(masks[i]) if i < len(masks) else 0
        axes[1].text(x1, y1-5, f"{class_name}\n{scores[i]:.2f} | {surface:,} px",
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color_norm, alpha=0.8))
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"YOLO26-seg ({len(boxes)} objets) | ⏱️ {format_time(inference_time)}")
    axes[1].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def export_masks(predictions, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(predictions['masks'])):
        class_name = predictions['class_names'][i]
        score = predictions['scores'][i]
        mask = (predictions['masks'][i] > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(output_dir, f"{i:02d}_{class_name}_{score:.2f}.png"))


def generate_report(predictions, image_name):
    inference_time = predictions.get('inference_time', 0)
    
    report = {
        'image': image_name,
        'timestamp': datetime.now().isoformat(),
        'inference_time_ms': inference_time * 1000,
        'total_objects': len(predictions['boxes']),
        'surfaces_by_class': {c: {'count': 0, 'total_surface_px': 0} for c in CLASSES[1:]},
        'details': []
    }
    
    for i in range(len(predictions['boxes'])):
        class_name = predictions['class_names'][i]
        score = float(predictions['scores'][i])
        surface = calculate_surface(predictions['masks'][i]) if i < len(predictions['masks']) else 0
        
        report['surfaces_by_class'][class_name]['count'] += 1
        report['surfaces_by_class'][class_name]['total_surface_px'] += surface
        report['details'].append({
            'id': i, 'class': class_name, 'score': score,
            'surface_px': surface, 'bbox': predictions['boxes'][i].tolist()
        })
    return report


# =============================================================================
# RÉSUMÉ GLOBAL
# =============================================================================

def generate_summary(all_reports, output_dir, total_processing_time):
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'YOLO26-seg',
        'total_images': len(all_reports),
        'total_processing_time_s': total_processing_time,
        'avg_inference_time_ms': 0,
        'total_objects': 0,
        'objects_by_class': {c: 0 for c in CLASSES[1:]},
        'surfaces_by_class': {c: 0 for c in CLASSES[1:]},
        'per_image_stats': []
    }
    
    total_inference_time = 0
    for report in all_reports:
        total_inference_time += report['inference_time_ms']
        summary['total_objects'] += report['total_objects']
        for class_name, data in report['surfaces_by_class'].items():
            summary['objects_by_class'][class_name] += data['count']
            summary['surfaces_by_class'][class_name] += data['total_surface_px']
        summary['per_image_stats'].append({
            'image': report['image'],
            'objects': report['total_objects'],
            'inference_time_ms': report['inference_time_ms']
        })
    
    summary['avg_inference_time_ms'] = total_inference_time / len(all_reports) if all_reports else 0
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    total_surface = sum(summary['surfaces_by_class'].values())
    with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RÉSUMÉ D'INFÉRENCE - YOLO26-SEG CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"📅 Date: {summary['timestamp']}\n")
        f.write(f"🖼️  Images traitées: {summary['total_images']}\n")
        f.write(f"⏱️  Temps total: {format_time(summary['total_processing_time_s'])}\n")
        f.write(f"⏱️  Temps moyen/image: {summary['avg_inference_time_ms']:.1f} ms\n")
        f.write(f"🎯 Total objets: {summary['total_objects']}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Classe':<25} {'Objets':>10} {'Surface (px)':>15} {'%':>10}\n")
        f.write("-" * 70 + "\n")
        for class_name in CLASSES[1:]:
            count = summary['objects_by_class'][class_name]
            surface = summary['surfaces_by_class'][class_name]
            pct = (surface / total_surface * 100) if total_surface > 0 else 0
            f.write(f"{class_name:<25} {count:>10} {surface:>15,} {pct:>9.1f}%\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<25} {summary['total_objects']:>10} {total_surface:>15,} {'100.0%':>10}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("DÉTAILS PAR IMAGE\n" + "-" * 70 + "\n")
        f.write(f"{'Image':<40} {'Objets':>10} {'Temps (ms)':>15}\n")
        f.write("-" * 70 + "\n")
        for stat in summary['per_image_stats']:
            img_name = stat['image'][:38] + '..' if len(stat['image']) > 40 else stat['image']
            f.write(f"{img_name:<40} {stat['objects']:>10} {stat['inference_time_ms']:>15.1f}\n")
        f.write("=" * 70 + "\n")
    
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("   📊 RÉSUMÉ GLOBAL - YOLO26-SEG")
    print("=" * 70)
    print(f"\n   🖼️  Images traitées:     {summary['total_images']}")
    print(f"   ⏱️  Temps total:          {format_time(summary['total_processing_time_s'])}")
    print(f"   ⏱️  Temps moyen/image:    {summary['avg_inference_time_ms']:.1f} ms")
    print(f"   🎯 Total objets:         {summary['total_objects']}")
    
    total_surface = sum(summary['surfaces_by_class'].values())
    print(f"\n   📋 Par classe:")
    for class_name in CLASSES[1:]:
        count = summary['objects_by_class'][class_name]
        surface = summary['surfaces_by_class'][class_name]
        pct = (surface / total_surface * 100) if total_surface > 0 else 0
        if count > 0:
            print(f"      • {class_name}: {count} objets | {surface:,} px ({pct:.1f}%)")
    print("\n" + "=" * 70)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, threshold=0.5, export_masks_flag=False, show_display=False):
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        return []
    
    print(f"\n🖼️  {len(image_paths)} images à traiter\n")
    
    all_reports = []
    start_total = time.time()
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 🔍 {img_path.name}")
        
        image, predictions = predict(model, str(img_path), threshold)
        
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=show_display)
        
        if export_masks_flag and len(predictions['masks']) > 0:
            export_masks(predictions, os.path.join(output_dir, "masks", img_path.stem), img_path.stem)
        
        report = generate_report(predictions, img_path.name)
        all_reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")
    
    total_processing_time = time.time() - start_total
    
    with open(os.path.join(output_dir, "reports.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    summary = generate_summary(all_reports, output_dir, total_processing_time)
    print_summary(summary)
    
    print(f"\n📁 Résultats sauvegardés dans: {output_dir}")
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Configuration depuis variables d'environnement
    model_path = CONFIG["model_path"]
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    score_threshold = CONFIG["score_threshold"]
    export_masks_flag = CONFIG["export_masks"]
    show_display = CONFIG["show_display"]
    
    # Vérifications
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print(f"   Définissez SEGMENTATION_MODEL_PATH")
        return
    
    if not os.path.exists(input_dir):
        print(f"❌ Dossier d'images non trouvé: {input_dir}")
        print(f"   Définissez SEGMENTATION_TEST_IMAGES_DIR")
        return
    
    print("=" * 70)
    print("   🚀 INFÉRENCE YOLO26-SEG CADASTRAL")
    print("=" * 70)
    print(f"\n📂 Configuration:")
    print(f"   • Modèle:      {model_path}")
    print(f"   • Images:      {input_dir}")
    print(f"   • Sortie:      {output_dir}")
    print(f"   • Seuil:       {score_threshold}")
    
    model = load_model(model_path)
    
    input_path = Path(input_dir)
    
    if input_path.is_dir():
        process_directory(model, str(input_path), output_dir, score_threshold, export_masks_flag, show_display)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🔍 Traitement: {input_path.name}")
        
        image, predictions = predict(model, str(input_path), score_threshold)
        
        output_path = os.path.join(output_dir, f"{input_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=show_display)
        
        if export_masks_flag and len(predictions['masks']) > 0:
            export_masks(predictions, os.path.join(output_dir, "masks"), input_path.stem)
        
        report = generate_report(predictions, input_path.name)
        print(f"\n{'='*60}")
        print(f"📊 RAPPORT - {report['image']}")
        print(f"{'='*60}")
        print(f"   ⏱️  Temps d'inférence: {report['inference_time_ms']:.1f} ms")
        print(f"   🎯 Objets détectés: {report['total_objects']}")
        for class_name, data in report['surfaces_by_class'].items():
            if data['count'] > 0:
                print(f"      • {class_name}: {data['count']} objets, {data['total_surface_px']:,} px")
        print(f"{'='*60}")
        
        with open(os.path.join(output_dir, f"{input_path.stem}_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()