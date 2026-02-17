# YOLO26-seg - Segmentation des Toitures Cadastrales

Projet de segmentation d'instance avec YOLO26 pour la classification automatique des types de toitures.
**Structure identique à Mask R-CNN et DeepLabV3+ pour comparaison équitable.**

## Avantages de YOLO26-seg

- **NMS-Free** : Inférence end-to-end sans post-traitement
- **43% plus rapide sur CPU** : Optimisé pour l'edge computing
- **State-of-the-art** : Dernière version YOLO (janvier 2026)
- **Segmentation d'instance** : Comme Mask R-CNN

## Structure des trois projets

```
maskrcnn_cadastral/          deeplab_cadastral/           yolo26_cadastral/
├── train.py                 ├── train.py                 ├── train.py
├── evaluate.py              ├── evaluate.py              ├── evaluate.py
├── inference.py             ├── inference.py             ├── inference.py
├── verify_dataset.py        ├── verify_dataset.py        ├── verify_dataset.py
├── requirements.txt         ├── requirements.txt         ├── requirements.txt
└── README.md                └── README.md                └── README.md
```

## Comparaison des modèles

| Aspect | Mask R-CNN | DeepLabV3+ | YOLO26-seg |
|--------|------------|------------|------------|
| **Type** | Instance seg. | Semantic seg. | Instance seg. |
| **Backbone** | ResNet50+FPN | ResNet50+ASPP | CSPDarknet |
| **NMS** | Requis | N/A | **Non requis** |
| **Vitesse** | Lent | Moyen | **Rapide** |
| **Edge-ready** | Non | Non | **Oui** |

## Métriques identiques pour comparaison

Les trois modèles sont évalués avec **exactement les mêmes métriques** :

| Métrique | Description |
|----------|-------------|
| mAP@50 | Mean Average Precision à IoU=0.5 |
| mAP@50:95 | Moyenne des AP de 0.5 à 0.95 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |
| IoU moyen | Intersection over Union |

## Installation

```bash
pip install -r requirements.txt
```

**Note:** YOLO26 nécessite `ultralytics>=8.4.0`

## Utilisation

### 1. Vérifier le dataset
```bash
python verify_dataset.py --images chemin/images --annotations chemin/annotations.json
```

### 2. Entraîner
```bash
python train.py
```

Le script convertit automatiquement le dataset COCO en format YOLO.

### 3. Évaluer
```bash
python evaluate.py
```

### 4. Inférence
```bash
python inference.py --model output/best_model.pt --input image.jpg
```

## Configuration

Modifier `train.py` :

```python
CONFIG = {
    "images_dir": "chemin/vers/images",
    "annotations_file": "chemin/vers/annotations.json",
    "output_dir": "./output",
    
    "model_size": "n",  # n, s, m, l, x
    "num_epochs": 25,
    "batch_size": 2,
    "image_size": 640,
    ...
}
```

### Tailles de modèle YOLO26-seg

| Modèle | Params | mAPmask | Vitesse CPU |
|--------|--------|---------|-------------|
| yolo26n-seg | 2.7M | 33.9 | ⚡ Très rapide |
| yolo26s-seg | 10.4M | 40.0 | ⚡ Rapide |
| yolo26m-seg | 23.6M | 44.1 | 🔹 Moyen |
| yolo26l-seg | 28.0M | 45.5 | 🔹 Moyen |
| yolo26x-seg | 62.8M | 47.0 | 🐢 Lent |

## Tableau de comparaison pour ta thèse

Après entraînement des trois modèles :

```
┌──────────────────┬────────────┬─────────────┬─────────────┐
│ Métrique         │ Mask R-CNN │ DeepLabV3+  │ YOLO26-seg  │
├──────────────────┼────────────┼─────────────┼─────────────┤
│ mAP@50           │            │             │             │
│ mAP@50:95        │            │             │             │
│ Precision@50     │            │             │             │
│ Recall@50        │            │             │             │
│ F1-Score@50      │            │             │             │
│ IoU moyen        │            │             │             │
│ Temps total      │            │             │             │
│ Temps/epoch      │            │             │             │
│ Paramètres       │   ~44M     │   ~40M      │   ~2.7-63M  │
└──────────────────┴────────────┴─────────────┴─────────────┘
```

## Fichiers générés

### Entraînement (output/)
```
output/
├── best_model.pt
├── final_model.pt
├── history.json
├── training_report.txt
├── dataset/           # Dataset converti en format YOLO
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
└── train/             # Résultats Ultralytics
    ├── weights/
    ├── results.csv
    └── *.png
```

### Évaluation (evaluation/)
```
evaluation/
├── metrics.json
├── evaluation_report.txt
├── metrics_per_class.png
└── metrics_vs_iou.png
```

## Auteur

Projet de thèse - Exploitation de l'IA pour l'évaluation cadastrale automatisée
Burkina Faso - SYCAD/DGI
