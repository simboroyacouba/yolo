"""
Évaluation complète du modèle YOLO26-seg
Métriques IDENTIQUES à Mask R-CNN et DeepLabV3+ pour comparaison équitable:
- mAP@50 (IoU threshold = 0.5)
- mAP@50:95 (IoU thresholds de 0.5 à 0.95)
- Precision, Recall, F1-Score
- IoU moyen
"""

import os
import json
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
import yaml
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN et DeepLabV3+)
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['classes']  # ✅ Ne pas filtrer __background__

CONFIG = {
    # Chemins
    "images_dir":  os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "model_path": "./runs/segment/output/train/weights/best.pt",
    "output_dir": "./evaluation",
    
    # Classes (identique aux autres modèles)
    "classes": load_classes(), 
    
    # Paramètres d'évaluation (identique aux autres modèles)
    "score_threshold": 0.5,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "image_size": 640,
}


# =============================================================================
# CALCUL DES MÉTRIQUES (identique à Mask R-CNN)
# =============================================================================

def calculate_iou_masks(mask1, mask2):
    """Calculer IoU entre deux masques binaires"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def calculate_iou_boxes(box1, box2):
    """Calculer IoU entre deux boîtes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(recalls, precisions):
    """Calculer AP = aire sous la courbe Precision-Recall (interpolation continue)."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return float(ap)


class MetricsCalculator:
    """
    Calcule le vrai mAP (mean Average Precision) = moyenne de l'AP
    (aire sous la courbe Precision-Recall) par classe et par seuil IoU.
    """

    def __init__(self, num_classes, class_names, iou_thresholds):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        # class_id -> liste de {'score', 'ious': np.array(n_gt,), 'img_idx'}
        self.detections = defaultdict(list)
        # class_id -> nombre total d'instances GT
        self.n_gts = defaultdict(int)
        self._img_idx = 0
        self.box_ious = []
        self.mask_ious = []

    def add_image(self, pred_boxes, pred_labels, pred_scores, pred_masks,
                  gt_boxes, gt_labels, gt_masks):
        """Ajouter une image pour évaluation."""
        for class_id in range(1, self.num_classes):
            pred_cls = pred_labels == class_id
            gt_cls = gt_labels == class_id

            pred_b = pred_boxes[pred_cls]
            pred_s = pred_scores[pred_cls]
            pred_m = [pred_masks[i] for i, v in enumerate(pred_cls) if v]
            gt_b = gt_boxes[gt_cls]
            gt_m = [gt_masks[i] for i, v in enumerate(gt_cls) if v]

            n_pred = len(pred_b)
            n_gt = len(gt_b)
            self.n_gts[class_id] += n_gt

            if n_pred == 0:
                continue

            # Matrice IoU (n_pred x n_gt)
            iou_matrix = np.zeros((n_pred, n_gt))
            for i in range(n_pred):
                for j in range(n_gt):
                    box_iou_val = calculate_iou_boxes(pred_b[i], gt_b[j])
                    if pred_m and gt_m:
                        mask_iou_val = calculate_iou_masks(pred_m[i], gt_m[j])
                        iou_matrix[i, j] = (box_iou_val + mask_iou_val) / 2
                        self.mask_ious.append(mask_iou_val)
                    else:
                        iou_matrix[i, j] = box_iou_val
                    self.box_ious.append(box_iou_val)

            # Stocker chaque détection avec son score et ses IoU avec les GT
            for i in range(n_pred):
                self.detections[class_id].append({
                    'score': float(pred_s[i]),
                    'ious': iou_matrix[i].copy(),
                    'img_idx': self._img_idx
                })

        self._img_idx += 1

    def _compute_ap(self, class_id, iou_thresh):
        """AP = aire sous la courbe PR pour une classe à un seuil IoU donné."""
        n_gt = self.n_gts[class_id]
        dets = self.detections[class_id]
        if n_gt == 0 or not dets:
            return 0.0

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)  # img_idx -> ensemble d'indices GT matchés
        tp_list, fp_list = [], []

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp_list.append(1); fp_list.append(0)
                matched[img_idx].add(best_j)
            else:
                tp_list.append(0); fp_list.append(1)

        tp_cum = np.cumsum(tp_list, dtype=float)
        fp_cum = np.cumsum(fp_list, dtype=float)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)
        return calculate_ap(recalls, precisions)

    def _compute_prf(self, class_id, iou_thresh, score_thresh=0.5):
        """TP/FP/FN et Precision/Recall/F1 à un seuil de score fixe."""
        n_gt = self.n_gts[class_id]
        dets = [d for d in self.detections[class_id] if d['score'] >= score_thresh]
        n_pred = len(dets)

        if n_gt == 0 and n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_gt == 0:
            return {'TP': 0, 'FP': n_pred, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': n_gt, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)
        tp = fp = 0

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp += 1; matched[img_idx].add(best_j)
            else:
                fp += 1

        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': precision, 'Recall': recall, 'F1': f1}

    def compute_metrics(self):
        """Calculer toutes les métriques finales."""
        results = {'per_class': {}, 'overall': {}, 'iou_stats': {}}
        score_thresh = CONFIG.get('score_threshold', 0.5)

        # --- AP et PRF par classe ---
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {}
            for iou_thresh in self.iou_thresholds:
                prf = self._compute_prf(class_id, iou_thresh, score_thresh)
                prf['AP'] = self._compute_ap(class_id, iou_thresh)
                results['per_class'][class_name][f'iou_{iou_thresh}'] = prf

        # --- Métriques globales micro-moyennées (à seuil fixe, pour le reporting) ---
        for iou_thresh in self.iou_thresholds:
            total_tp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['TP']
                           for c in range(1, self.num_classes))
            total_fp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FP']
                           for c in range(1, self.num_classes))
            total_fn = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FN']
                           for c in range(1, self.num_classes))
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results['overall'][f'iou_{iou_thresh}'] = {
                'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
                'Precision': precision, 'Recall': recall, 'F1': f1
            }

        # --- mAP@50 = moyenne de AP@50 sur toutes les classes ---
        results['mAP50'] = float(np.mean([
            results['per_class'][self.class_names[c]]['iou_0.5']['AP']
            for c in range(1, self.num_classes)
        ]))

        # --- mAP@50:95 = moyenne de AP sur toutes les classes ET tous les seuils ---
        results['mAP50_95'] = float(np.mean([
            results['per_class'][self.class_names[c]][f'iou_{t}']['AP']
            for c in range(1, self.num_classes)
            for t in self.iou_thresholds
        ]))

        # --- AP par classe ---
        results['mAP_per_class'] = {}
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['mAP_per_class'][class_name] = {
                'AP50': results['per_class'][class_name]['iou_0.5']['AP'],
                'AP50_95': float(np.mean([
                    results['per_class'][class_name][f'iou_{t}']['AP']
                    for t in self.iou_thresholds
                ]))
            }

        # --- Stats IoU ---
        if self.box_ious:
            results['iou_stats']['box_iou_mean'] = float(np.mean(self.box_ious))
            results['iou_stats']['box_iou_std'] = float(np.std(self.box_ious))
            results['iou_stats']['box_iou_median'] = float(np.median(self.box_ious))
        if self.mask_ious:
            results['iou_stats']['mask_iou_mean'] = float(np.mean(self.mask_ious))
            results['iou_stats']['mask_iou_std'] = float(np.std(self.mask_ious))
            results['iou_stats']['mask_iou_median'] = float(np.median(self.mask_ious))

        return results


# =============================================================================
# CHARGEMENT DES GROUND TRUTHS
# =============================================================================

def load_ground_truths(images_dir, annotations_file):
    """Charger les ground truths depuis COCO"""
    
    coco = COCO(annotations_file)
    
    cat_ids = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
    
    ground_truths = {}
    
    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_mapping[ann['category_id']])
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)
        
        ground_truths[img_info['file_name']] = {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
            'labels': np.array(labels) if labels else np.zeros((0,), dtype=int),
            'masks': masks,
            'image_id': img_id,
            'width': img_info['width'],
            'height': img_info['height']
        }
    
    return ground_truths


# =============================================================================
# VISUALISATION (identique aux autres modèles)
# =============================================================================

def plot_metrics(results, output_dir):
    """Créer les graphiques des métriques"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. AP par classe
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names = list(results['mAP_per_class'].keys())
    ap50_values = [results['mAP_per_class'][c]['AP50'] for c in class_names]
    ap50_95_values = [results['mAP_per_class'][c]['AP50_95'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0].bar(x - width/2, ap50_values, width, label='AP@50', color='steelblue')
    axes[0].bar(x + width/2, ap50_95_values, width, label='AP@50:95', color='coral')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('AP par classe')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Precision, Recall, F1
    precisions = [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names]
    recalls = [results['per_class'][c]['iou_0.5']['Recall'] for c in class_names]
    f1s = [results['per_class'][c]['iou_0.5']['F1'] for c in class_names]
    
    width = 0.25
    axes[1].bar(x - width, precisions, width, label='Precision', color='green')
    axes[1].bar(x, recalls, width, label='Recall', color='blue')
    axes[1].bar(x + width, f1s, width, label='F1-Score', color='red')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision / Recall / F1 par classe (IoU=0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_per_class.png'), dpi=150)
    plt.close()
    
    # 3. Métriques vs seuil IoU
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iou_thresholds = CONFIG['iou_thresholds']
    global_precisions = [results['overall'][f'iou_{t}']['Precision'] for t in iou_thresholds]
    global_recalls = [results['overall'][f'iou_{t}']['Recall'] for t in iou_thresholds]
    global_f1s = [results['overall'][f'iou_{t}']['F1'] for t in iou_thresholds]
    
    ax.plot(iou_thresholds, global_precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_recalls, 's-', label='Recall', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_f1s, '^-', label='F1-Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Seuil IoU')
    ax.set_ylabel('Score')
    ax.set_title('Métriques globales vs Seuil IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.45, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_iou.png'), dpi=150)
    plt.close()
    
    print(f"📊 Graphiques sauvegardés dans: {output_dir}")


def generate_report(results, output_dir):
    """Générer un rapport complet (identique aux autres modèles)"""
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ÉVALUATION - YOLO26-seg CADASTRAL\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📊 RÉSUMÉ DES MÉTRIQUES PRINCIPALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)\n")
        f.write(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}\n")
        
        if results.get('iou_stats'):
            f.write(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}\n")
            f.write(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}\n")
        
        f.write("\n\n📋 MÉTRIQUES PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10}\n")
        f.write("-" * 65 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            ap50 = results['mAP_per_class'][class_name]['AP50']
            f.write(f"{class_name:<25} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
                   f"{metrics['F1']:>10.4f} {ap50:>10.4f}\n")
        
        f.write("\n\n📈 DÉTAILS TP/FP/FN PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'TP':>8} {'FP':>8} {'FN':>8}\n")
        f.write("-" * 50 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            f.write(f"{class_name:<25} {metrics['TP']:>8} {metrics['FP']:>8} {metrics['FN']:>8}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"📄 Rapport sauvegardé: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   ÉVALUATION YOLO26-seg - Segmentation des Toitures")
    print("   (Métriques identiques à Mask R-CNN et DeepLabV3+)")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Charger les ground truths
    print("\n📂 Chargement des ground truths...")
    ground_truths = load_ground_truths(
        CONFIG["images_dir"],
        CONFIG["annotations_file"]
    )
    print(f"   {len(ground_truths)} images chargées")
    
    # Charger le modèle
    print("\n🧠 Chargement du modèle...")
    model = YOLO(CONFIG["model_path"])
    
    # Initialiser le calculateur de métriques
    num_classes = len(CONFIG["classes"])
    metrics_calc = MetricsCalculator(
        num_classes=num_classes,
        class_names=CONFIG["classes"],
        iou_thresholds=CONFIG["iou_thresholds"]
    )
    
    # Évaluation
    print("\n📊 Calcul des métriques...")
    
    image_files = list(ground_truths.keys())
    
    for img_file in tqdm(image_files, desc="Évaluation"):
        img_path = os.path.join(CONFIG["images_dir"], img_file)
        
        if not os.path.exists(img_path):
            continue
        
        # Prédiction — conf=0.001 pour inclure toutes les détections
        # et construire la courbe PR sur toute la plage de confiance
        results = model.predict(
            img_path,
            conf=0.001,
            imgsz=CONFIG["image_size"],
            verbose=False
        )
        
        result = results[0]
        gt = ground_truths[img_file]
        
        # Extraire les prédictions
        if result.masks is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_labels = result.boxes.cls.cpu().numpy().astype(int) + 1  # +1 car YOLO commence à 0
            pred_scores = result.boxes.conf.cpu().numpy()
            
            # Masques
            pred_masks = []
            masks_data = result.masks.data.cpu().numpy()
            for i in range(len(masks_data)):
                # Redimensionner le masque à la taille originale
                mask = masks_data[i]
                mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                    (gt['width'], gt['height']), Image.NEAREST
                ))
                pred_masks.append(mask_resized > 0.5)
        else:
            pred_boxes = np.zeros((0, 4))
            pred_labels = np.zeros((0,), dtype=int)
            pred_scores = np.zeros((0,))
            pred_masks = []
        
        # Ajouter pour calcul
        metrics_calc.add_image(
            pred_boxes, pred_labels, pred_scores, pred_masks,
            gt['boxes'], gt['labels'], gt['masks']
        )
    
    # Calculer les métriques
    results = metrics_calc.compute_metrics()
    
    # Affichage (identique aux autres modèles)
    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)
    
    print(f"\n🎯 MÉTRIQUES PRINCIPALES")
    print(f"   {'─' * 40}")
    print(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)")
    print(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}")
    
    if results.get('iou_stats'):
        print(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}")
        print(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}")
    
    print(f"\n📋 PAR CLASSE (IoU=0.5)")
    print(f"   {'─' * 40}")
    for class_name in results['per_class']:
        metrics = results['per_class'][class_name]['iou_0.5']
        print(f"   {class_name}:")
        print(f"      Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
    
    # Sauvegarder les résultats
    results_path = os.path.join(CONFIG["output_dir"], "metrics.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n💾 Métriques sauvegardées: {results_path}")
    
    # Graphiques
    plot_metrics(results, CONFIG["output_dir"])
    
    # Rapport
    generate_report(results, CONFIG["output_dir"])
    
    print("\n" + "=" * 70)
    print("   ✅ ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
