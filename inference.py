"""
Inférence YOLO26-seg - Prédiction sur nouvelles images
Structure identique à Mask R-CNN et DeepLabV3+ pour comparaison
"""

import os
import json
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique aux autres modèles)
# =============================================================================

CLASSES = [
    "__background__",
    "toiture_tole_ondulee",
    "toiture_tole_bac",
    "toiture_tuile",
    "toiture_dalle"
]

COLORS = {
    "toiture_tole_ondulee": (255, 0, 0),
    "toiture_tole_bac": (0, 255, 0),
    "toiture_tuile": (0, 0, 255),
    "toiture_dalle": (255, 165, 0),
}


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, score_threshold=0.5):
    """Faire une prédiction sur une image"""
    
    image = Image.open(image_path).convert("RGB")
    
    results = model.predict(
        image_path,
        conf=score_threshold,
        verbose=False
    )
    
    result = results[0]
    
    predictions = {
        'boxes': [],
        'labels': [],
        'scores': [],
        'masks': [],
        'class_names': []
    }
    
    if result.masks is not None and len(result.boxes) > 0:
        predictions['boxes'] = result.boxes.xyxy.cpu().numpy()
        predictions['labels'] = result.boxes.cls.cpu().numpy().astype(int)
        predictions['scores'] = result.boxes.conf.cpu().numpy()
        
        # Masques
        masks_data = result.masks.data.cpu().numpy()
        orig_shape = result.orig_shape
        
        for i in range(len(masks_data)):
            mask = masks_data[i]
            mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (orig_shape[1], orig_shape[0]), Image.NEAREST
            ))
            predictions['masks'].append(mask_resized > 0.5)
        
        predictions['class_names'] = [CLASSES[int(l) + 1] for l in predictions['labels']]
    
    return image, predictions


def calculate_surface(mask, pixel_size_m2=None):
    """Calculer la surface d'un masque"""
    surface_pixels = np.sum(mask > 0)
    
    if pixel_size_m2 is not None:
        return surface_pixels * pixel_size_m2
    return surface_pixels


# =============================================================================
# VISUALISATION (identique aux autres modèles)
# =============================================================================

def visualize_predictions(image, predictions, output_path=None, show=True):
    """Visualiser les prédictions avec masques et boîtes"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Image avec prédictions
    axes[1].imshow(image)
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions['masks']
    class_names = predictions['class_names']
    
    # Overlay des masques
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for i in range(len(boxes)):
        class_name = class_names[i]
        color = COLORS.get(class_name, (128, 128, 128))
        color_normalized = [c/255 for c in color]
        
        # Masque
        if i < len(masks):
            mask = masks[i]
            overlay[mask > 0] = [*color_normalized, 0.5]
        
        # Boîte
        box = boxes[i]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=color_normalized,
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Label
        score = scores[i]
        surface = calculate_surface(masks[i]) if i < len(masks) else 0
        label_text = f"{class_name}\n{score:.2f} | {surface:,} px"
        axes[1].text(
            x1, y1-10,
            label_text,
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor=color_normalized, alpha=0.8)
        )
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prédictions ({len(boxes)} objets détectés)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Résultat sauvegardé: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def export_masks(predictions, output_dir, image_name):
    """Exporter les masques individuels"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (mask, class_name, score) in enumerate(zip(
        predictions['masks'],
        predictions['class_names'],
        predictions['scores']
    )):
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_binary)
        
        mask_path = os.path.join(
            output_dir,
            f"{image_name}_{i:02d}_{class_name}_{score:.2f}.png"
        )
        mask_image.save(mask_path)
    
    print(f"Masques exportés dans: {output_dir}")


def generate_report(predictions, image_name):
    """Générer un rapport des surfaces détectées (identique aux autres modèles)"""
    
    report = {
        'image': image_name,
        'total_objects': len(predictions['boxes']),
        'surfaces_by_class': {},
        'details': []
    }
    
    for class_name in CLASSES[1:]:
        report['surfaces_by_class'][class_name] = {
            'count': 0,
            'total_surface_px': 0
        }
    
    for i in range(len(predictions['boxes'])):
        class_name = predictions['class_names'][i]
        score = predictions['scores'][i]
        box = predictions['boxes'][i]
        surface = calculate_surface(predictions['masks'][i]) if i < len(predictions['masks']) else 0
        
        report['surfaces_by_class'][class_name]['count'] += 1
        report['surfaces_by_class'][class_name]['total_surface_px'] += surface
        
        report['details'].append({
            'id': i,
            'class': class_name,
            'score': float(score),
            'surface_px': int(surface),
            'bbox': box.tolist()
        })
    
    return report


def print_report(report):
    """Afficher le rapport (identique aux autres modèles)"""
    print("\n" + "=" * 50)
    print(f"RAPPORT DE SEGMENTATION - {report['image']}")
    print("=" * 50)
    print(f"Total objets détectés: {report['total_objects']}")
    print("\nSurfaces par classe:")
    print("-" * 50)
    
    for class_name, data in report['surfaces_by_class'].items():
        if data['count'] > 0:
            print(f"  {class_name}:")
            print(f"    - Nombre: {data['count']}")
            print(f"    - Surface totale: {data['total_surface_px']:,} pixels")
    
    print("\nDétails:")
    print("-" * 50)
    for obj in report['details']:
        print(f"  [{obj['id']}] {obj['class']} (conf: {obj['score']:.2f}) - {obj['surface_px']:,} px")
    
    print("=" * 50)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, score_threshold=0.5):
    """Traiter toutes les images d'un répertoire"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_paths = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"\nTraitement de {len(image_paths)} images...")
    
    all_reports = []
    
    for img_path in tqdm(image_paths, desc="Inférence"):
        # Prédiction
        image, predictions = predict(model, str(img_path), score_threshold)
        
        # Visualisation
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=False)
        
        # Rapport
        report = generate_report(predictions, img_path.name)
        all_reports.append(report)
        print_report(report)
    
    # Sauvegarder tous les rapports
    reports_path = os.path.join(output_dir, "reports.json")
    with open(reports_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\nRapports sauvegardés: {reports_path}")
    
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence YOLO26-seg Cadastral")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle")
    parser.add_argument("--input", type=str, required=True, help="Image ou dossier d'images")
    parser.add_argument("--output", type=str, default="./predictions", help="Dossier de sortie")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de confiance")
    parser.add_argument("--export-masks", action="store_true", help="Exporter les masques individuels")
    
    args = parser.parse_args()
    
    # Charger le modèle
    print(f"Chargement du modèle: {args.model}")
    model = YOLO(args.model)
    
    # Traitement
    input_path = Path(args.input)
    
    if input_path.is_dir():
        process_directory(model, str(input_path), args.output, args.threshold)
    else:
        os.makedirs(args.output, exist_ok=True)
        
        image, predictions = predict(model, str(input_path), args.threshold)
        
        output_path = os.path.join(args.output, f"{input_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path)
        
        if args.export_masks:
            export_masks(predictions, os.path.join(args.output, "masks"), input_path.stem)
        
        report = generate_report(predictions, input_path.name)
        print_report(report)


if __name__ == "__main__":
    main()
