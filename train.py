"""
Entraînement YOLO26-seg pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)

Modes d'entraînement:
  simple    : YOLO26-seg standard, hyperparamètres fixes
  attention : YOLO26-seg + CBAM injecté sur les sorties neck avant la tête Detect
  optimize  : YOLO26-seg standard + recherche bayésienne des hyperparamètres (Optuna)
"""

import os
import json
import yaml
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import gc
from collections import OrderedDict
from PIL import Image
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [c for c in data['classes'] if c != '__background__']

OPTUNA_CONFIG = {
    "n_trials": 30,
    "n_epochs_per_trial": 5,
    "study_name": "yolo26_cadastral",
    "output_dir": "./optuna_output",
}

CONFIG = {
    "images_dir":       os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file":     os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir":       "./output",
    "classes":          load_classes(),
    "model_size":       "n",        # n, s, m, l, x
    "num_epochs":       25,
    "batch_size":       2,
    "learning_rate":    0.005,
    "momentum":         0.9,
    "weight_decay":     0.0005,
    "image_size":       640,
    "train_split":      0.85,
    "save_every":       5,
    # Paramètres CBAM par défaut (mode attention uniquement)
    "cbam_reduction":   16,
    "cbam_kernel_size": 7,
}


# =============================================================================
# UTILITAIRES TEMPS
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — injecté en mode "attention"
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# =============================================================================
# AUGMENTATION PAR CLASSE
# =============================================================================

def load_aug_coefficients(classes):
    """
    Charge les coefficients d'augmentation depuis les variables d'environnement.
    Format index  : CLASS_AUG_1=2  (1 = première classe)
    Format nom    : CLASS_AUG_TOITURE_TOLE_BAC=2
    Valeur par défaut: 1 (aucune augmentation).
    """
    real_classes = [c for c in classes if c != '__background__']
    coeffs = {}
    for i, cls in enumerate(real_classes, 1):
        env_idx  = f"CLASS_AUG_{i}"
        env_name = "CLASS_AUG_" + cls.upper().replace(' ', '_').replace('-', '_')
        raw = os.getenv(env_name) or os.getenv(env_idx, "1")
        coeffs[cls] = max(1, int(raw))
    return coeffs


def _count_yolo_class_stats(coco, image_ids, cat_id_to_name):
    stats = {name: {'images': set(), 'annotations': 0} for name in cat_id_to_name.values()}
    for img_id in image_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                name = cat_id_to_name[cid]
                stats[name]['images'].add(img_id)
                stats[name]['annotations'] += 1
    return {name: {'images': len(s['images']), 'annotations': s['annotations']}
            for name, s in stats.items()}


def _get_img_max_coeff(image_ids, coco, cat_id_to_name, aug_coeffs):
    """Retourne dict img_id -> coefficient maximal parmi ses classes."""
    result = {}
    for img_id in image_ids:
        max_c = 1
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                max_c = max(max_c, aug_coeffs.get(cat_id_to_name[cid], 1))
        result[img_id] = max_c
    return result


def _write_augmented_copy(src_img, src_lbl, dst_img, dst_lbl):
    """
    Crée une copie augmentée (flips aléatoires) d'une image YOLO.
    Met à jour les coordonnées des polygones dans le fichier label.
    """
    img = Image.open(src_img)
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(dst_img)

    with open(src_lbl, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            new_lines.append(line)
            continue
        cls_id = parts[0]
        coords = [float(p) for p in parts[1:]]
        new_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i + 1]
            if flip_h:
                x = 1.0 - x
            if flip_v:
                y = 1.0 - y
            new_coords.extend([x, y])
        new_lines.append(cls_id + " " + " ".join(f"{c:.6f}" for c in new_coords) + "\n")
    with open(dst_lbl, 'w') as f:
        f.writelines(new_lines)


def print_augmentation_report_yolo(coco, train_ids_before, augmented_count,
                                    cat_id_to_name, aug_coeffs):
    before = _count_yolo_class_stats(coco, train_ids_before, cat_id_to_name)
    print(f"\n{'='*70}")
    print(f"   RAPPORT D'AUGMENTATION DES DONNEES D'ENTRAINEMENT (YOLO)")
    print(f"{'='*70}")
    print(f"\n   AVANT AUGMENTATION  ({len(set(train_ids_before))} images uniques en train)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Coeff':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_b = 0
    for cls_name, s in before.items():
        coeff  = aug_coeffs.get(cls_name, 1)
        marker = "  *" if coeff > 1 else ""
        print(f"   {cls_name:<38} x{coeff:>4}  {s['images']:>7}  {s['annotations']:>7}{marker}")
        total_ann_b += s['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL':<38}       {len(set(train_ids_before)):>7}  {total_ann_b:>7}")
    print(f"\n   APRES AUGMENTATION  ({augmented_count} fichiers images dans train/)")
    ratio = augmented_count / max(len(train_ids_before), 1)
    print(f"   Ratio d'augmentation global: x{ratio:.2f} samples")
    print(f"{'='*70}\n")


# =============================================================================
# CONVERSION COCO -> YOLO FORMAT
# =============================================================================

def coco_to_yolo_segmentation(ann, img_width, img_height):
    segmentation = ann.get('segmentation', [])
    if not segmentation or not isinstance(segmentation, list):
        return None
    polygon = segmentation[0] if segmentation else []
    if len(polygon) < 6:
        return None
    normalized = []
    for i in range(0, len(polygon), 2):
        x = max(0.0, min(1.0, polygon[i]     / img_width))
        y = max(0.0, min(1.0, polygon[i + 1] / img_height))
        normalized.extend([x, y])
    return normalized


def prepare_yolo_dataset(images_dir, annotations_file, output_dir, classes,
                         train_split=0.85, aug_coeffs=None):
    print("Preparation du dataset YOLO...")
    dataset_dir      = os.path.join(output_dir, "dataset")
    train_images_dir = os.path.join(dataset_dir, "images", "train")
    val_images_dir   = os.path.join(dataset_dir, "images", "val")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    val_labels_dir   = os.path.join(dataset_dir, "labels", "val")
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)

    coco        = COCO(annotations_file)
    cat_ids     = coco.getCatIds()
    cat_mapping = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    cat_id_to_name = {cat_id: coco.cats[cat_id]['name'] for cat_id in cat_ids}
    image_ids   = list(coco.imgs.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_split)
    train_ids, val_ids = image_ids[:split_idx], image_ids[split_idx:]

    if aug_coeffs is None:
        aug_coeffs = {name: 1 for name in cat_id_to_name.values()}

    # Coefficient max par image train
    img_max_coeff = _get_img_max_coeff(train_ids, coco, cat_id_to_name, aug_coeffs)
    total_train_files = sum(img_max_coeff.values())
    print(f"   Train: {len(train_ids)} images  ->  {total_train_files} samples apres augmentation")
    print(f"   Val:   {len(val_ids)} images")

    stats = {'train': 0, 'val': 0, 'annotations': 0}

    # ── Validation (copie simple) ──────────────────────────────────────────
    for img_id in val_ids:
        img_info = coco.imgs[img_id]
        src_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(src_path):
            continue
        shutil.copy2(src_path, os.path.join(val_images_dir, img_info['file_name']))
        stem      = os.path.splitext(img_info['file_name'])[0]
        lbl_path  = os.path.join(val_labels_dir, stem + '.txt')
        anns      = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        with open(lbl_path, 'w') as f:
            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue
                class_id = cat_mapping.get(ann['category_id'])
                polygon  = coco_to_yolo_segmentation(ann, img_info['width'], img_info['height'])
                if class_id is None or polygon is None:
                    continue
                f.write(f"{class_id} " + " ".join(f"{c:.6f}" for c in polygon) + "\n")
                stats['annotations'] += 1
        stats['val'] += 1

    # ── Train (original + copies augmentées) ──────────────────────────────
    for img_id in train_ids:
        img_info = coco.imgs[img_id]
        src_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(src_path):
            continue

        stem     = os.path.splitext(img_info['file_name'])[0]
        ext      = os.path.splitext(img_info['file_name'])[1]
        anns     = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        coeff    = img_max_coeff.get(img_id, 1)

        # Copie originale
        dst_img = os.path.join(train_images_dir, img_info['file_name'])
        dst_lbl = os.path.join(train_labels_dir, stem + '.txt')
        shutil.copy2(src_path, dst_img)
        with open(dst_lbl, 'w') as f:
            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue
                class_id = cat_mapping.get(ann['category_id'])
                polygon  = coco_to_yolo_segmentation(ann, img_info['width'], img_info['height'])
                if class_id is None or polygon is None:
                    continue
                f.write(f"{class_id} " + " ".join(f"{c:.6f}" for c in polygon) + "\n")
                stats['annotations'] += 1
        stats['train'] += 1

        # Copies augmentées (coeff - 1)
        for k in range(1, coeff):
            aug_name = f"{stem}_aug{k}{ext}"
            aug_img  = os.path.join(train_images_dir, aug_name)
            aug_lbl  = os.path.join(train_labels_dir, f"{stem}_aug{k}.txt")
            _write_augmented_copy(src_path, dst_lbl, aug_img, aug_lbl)
            stats['train'] += 1

    # Rapport d'augmentation
    print_augmentation_report_yolo(
        coco, train_ids, stats['train'], cat_id_to_name, aug_coeffs
    )

    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path':  os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val':   'images/val',
            'names': {i: name for i, name in enumerate(classes)},
        }, f, default_flow_style=False)

    print(f"   Annotations converties: {stats['annotations']}")
    return yaml_path, stats


# =============================================================================
# MODÈLE
# =============================================================================

def get_model_simple():
    """YOLO26-seg standard."""
    return YOLO(f"yolo26{CONFIG['model_size']}-seg.pt")


def get_model_attention(cbam_reduction=16, cbam_kernel_size=7):
    """YOLO26-seg standard — CBAM sera injecté via le custom trainer."""
    return get_model_simple()


def _make_cbam_trainer(cbam_reduction=16, cbam_kernel_size=7):
    """
    Retourne une classe Trainer qui injecte CBAM dans get_model(), APRES que
    Ultralytics recharge le modèle depuis le .pt. Cela garantit que les hooks
    et les paramètres CBAM sont bien présents lors de la construction de l'optimiseur.
    """
    from ultralytics.models.yolo.segment import SegmentationTrainer as _Base

    def _get_model(self, cfg=None, weights=None, verbose=True):
        model = _Base.get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        _inject_cbam_nn(model, cbam_reduction, cbam_kernel_size)
        return model

    return type("CBAMSegTrainer", (_Base,), {"get_model": _get_model})


def _inject_cbam_nn(nn_model, cbam_reduction=16, cbam_kernel_size=7):
    """Injecte CBAM directement sur un nn.Module (DetectionModel/SegmentationModel)."""
    detect_layer = nn_model.model[-1]
    source_f     = detect_layer.f
    n            = len(nn_model.model)
    abs_indices  = [i if i >= 0 else n + i for i in source_f]

    channels     = {}
    temp_handles = []

    def make_capture(idx):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                channels[idx] = out.shape[1]
        return hook

    for idx in abs_indices:
        temp_handles.append(nn_model.model[idx].register_forward_hook(make_capture(idx)))

    device = next(nn_model.parameters()).device
    dummy  = torch.zeros(1, 3, 640, 640, device=device)
    with torch.no_grad():
        try:
            nn_model(dummy)
        except Exception:
            pass
    for h in temp_handles:
        h.remove()

    if not channels:
        print("   [AVERTISSEMENT] Channels CBAM non detectes — mode simple utilise.")
        return

    cbam_dict = nn.ModuleDict({
        str(idx): CBAM(c, cbam_reduction, cbam_kernel_size).to(device)
        for idx, c in channels.items()
    })
    nn_model.cbam_attention = cbam_dict

    def make_cbam_hook(cbam_mod):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                return cbam_mod(out)
        return hook

    for idx in channels:
        nn_model.model[idx].register_forward_hook(make_cbam_hook(cbam_dict[str(idx)]))

    print(f"   CBAM injecte sur {len(channels)} niveaux: {list(channels.values())} channels")


def _inject_cbam_yolo(yolo_model, cbam_reduction=16, cbam_kernel_size=7):
    """
    Injecte des modules CBAM via des forward hooks sur les couches qui alimentent
    la tête Detect. Les modules sont enregistrés comme sous-modules du réseau
    pour être inclus dans l'optimiseur.
    """
    nn_model     = yolo_model.model
    detect_layer = nn_model.model[-1]
    source_f     = detect_layer.f  # ex: [-3, -2, -1] ou [15, 18, 21]
    n            = len(nn_model.model)
    abs_indices  = [i if i >= 0 else n + i for i in source_f]

    # --- passe factice pour capturer les channels ---
    channels     = {}
    temp_handles = []

    def make_capture(idx):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                channels[idx] = out.shape[1]
        return hook

    for idx in abs_indices:
        temp_handles.append(nn_model.model[idx].register_forward_hook(make_capture(idx)))

    device = next(nn_model.parameters()).device
    dummy  = torch.zeros(1, 3, 640, 640, device=device)
    with torch.no_grad():
        try:
            nn_model(dummy)
        except Exception:
            pass
    for h in temp_handles:
        h.remove()

    if not channels:
        print("   [AVERTISSEMENT] Channels CBAM non detectes — mode simple utilise.")
        return

    # --- enregistrer les modules CBAM comme sous-modules ---
    cbam_dict = nn.ModuleDict({
        str(idx): CBAM(c, cbam_reduction, cbam_kernel_size).to(device)
        for idx, c in channels.items()
    })
    nn_model.cbam_attention = cbam_dict  # registré → params inclus dans l'optimiseur

    # --- hooks permanents ---
    def make_cbam_hook(cbam_mod):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                return cbam_mod(out)
        return hook

    for idx, c in channels.items():
        nn_model.model[idx].register_forward_hook(make_cbam_hook(cbam_dict[str(idx)]))

    print(f"   CBAM injecte sur {len(channels)} niveaux de features: {list(channels.values())} channels")


# =============================================================================
# OPTIMISATION BAYÉSIENNE (OPTUNA) — mode "optimize" uniquement
# =============================================================================

def _run_optimization(yaml_path):
    """Recherche bayésienne des hyperparamètres via Optuna (sans CBAM)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        import optuna
        lr0          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
        momentum     = trial.suggest_float("momentum",      0.80, 0.99)

        model = get_model_simple()
        gc.collect()

        # Callback appelé à chaque fin d'epoch pour permettre au pruner d'agir
        epoch_losses = []
        def on_fit_epoch_end(trainer):
            loss = trainer.metrics.get("val/seg_loss")
            if loss is not None:
                step = len(epoch_losses)
                epoch_losses.append(float(loss))
                trial.report(float(loss), step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=yaml_path,
            epochs=OPTUNA_CONFIG["n_epochs_per_trial"],
            batch=CONFIG["batch_size"],
            imgsz=CONFIG["image_size"],
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            project=os.path.join(OPTUNA_CONFIG["output_dir"], "trials"),
            name=f"trial_{trial.number}",
            verbose=False,
            plots=False,
            cache=False,
            workers=0,
            amp=False,
            exist_ok=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss = results.results_dict.get("val/seg_loss", float("inf"))
        return float(val_loss) if val_loss is not None else float("inf")

    os.makedirs(OPTUNA_CONFIG["output_dir"], exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=OPTUNA_CONFIG["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"\n{'=' * 70}")
    print(f"   OPTIMISATION BAYESIENNE — {OPTUNA_CONFIG['n_trials']} essais")
    print(f"   {OPTUNA_CONFIG['n_epochs_per_trial']} epochs/essai | sampler: TPE | pruner: Median")
    print(f"{'=' * 70}\n")

    study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"   MEILLEUR ESSAI #{best.number}  —  val_loss: {best.value:.4f}")
    print(f"{'=' * 70}")
    for k, v in best.params.items():
        print(f"   {k}: {v}")

    report = {
        "best_trial": best.number, "best_val_loss": best.value, "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    report_path = os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Rapport sauvegarde: {report_path}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        values = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(values, marker='o', linewidth=1.5)
        axes[0].set_xlabel("Essai"); axes[0].set_ylabel("Val Loss")
        axes[0].set_title("Historique Optuna"); axes[0].grid(True, alpha=0.3)
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys()), list(importances.values()))
        axes[1].set_xlabel("Importance"); axes[1].set_title("Importance des hyperparametres")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_results.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return best.params


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def _run_training(model, yaml_path, model_config, trainer_cls=None):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print("\n" + "=" * 70)
    print("   DEBUT DE L'ENTRAINEMENT")
    print(f"   Demarre le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}")
    print("=" * 70)

    start_time     = time.time()
    training_start = datetime.now()

    train_kwargs = dict(
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
        cache=False,
        workers=0,
        amp=False,
    )
    if trainer_cls is not None:
        train_kwargs["trainer"] = trainer_cls
    results = model.train(**train_kwargs)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_time    = time.time() - start_time
    training_end  = datetime.now()
    avg_epoch_t   = total_time / CONFIG["num_epochs"]

    history = {'train_loss': [], 'val_loss': [], 'mAP50': [], 'mAP50_95': [],
               'epoch_times': [avg_epoch_t] * CONFIG["num_epochs"]}

    # Ultralytics sauvegarde dans runs/segment/{project}/{name}/, pas dans {project}/{name}/
    train_dir   = str(results.save_dir)
    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_csv):
        import csv
        with open(results_csv, 'r') as f:
            for row in csv.DictReader(f):
                try:
                    history['train_loss'].append(float(row.get('train/seg_loss', 0) or 0))
                    history['val_loss'].append(  float(row.get('val/seg_loss',   0) or 0))
                    history['mAP50'].append(     float(row.get('metrics/mAP50(M)',    0) or 0))
                    history['mAP50_95'].append(  float(row.get('metrics/mAP50-95(M)', 0) or 0))
                except Exception:
                    pass

    time_stats = {
        'total_time': total_time,
        'total_time_formatted': format_time(total_time),
        'avg_epoch_time': avg_epoch_t,
        'avg_epoch_time_formatted': format_time(avg_epoch_t),
        'start_datetime': training_start.strftime("%Y-%m-%d %H:%M:%S"),
        'end_datetime':   training_end.strftime("%Y-%m-%d %H:%M:%S"),
    }
    history['time_stats']   = time_stats
    history['model_config'] = model_config

    # Copier les poids
    for src_name, dst_name in [("best.pt", "best_model.pt"), ("last.pt", "final_model.pt")]:
        src = os.path.join(train_dir, "weights", src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(CONFIG["output_dir"], dst_name))

    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    best_mAP50    = max(history['mAP50'])    if history['mAP50']    else 0.0
    best_mAP50_95 = max(history['mAP50_95']) if history['mAP50_95'] else 0.0

    print("\n" + "=" * 70)
    print("   ENTRAINEMENT TERMINE")
    print("=" * 70)
    print(f"   Meilleur mAP@50:    {best_mAP50:.4f}")
    print(f"   Meilleur mAP@50:95: {best_mAP50_95:.4f}")
    print(f"   Temps total:        {time_stats['total_time_formatted']}")
    print(f"   Temps moyen/epoch:  {time_stats['avg_epoch_time_formatted']}")
    print(f"   Fichiers:           {CONFIG['output_dir']}/")
    print("=" * 70)

    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ENTRAINEMENT - YOLO26-seg CADASTRAL\n")
        f.write("=" * 70 + "\n\nCONFIGURATION\n")
        for k, v in CONFIG.items():
            f.write(f"   {k}: {v}\n")
        f.write(f"\nMODE: {model_config.get('mode','?')}\n")
        f.write(f"\nPERFORMANCES\n   mAP@50: {best_mAP50:.4f}\n   mAP@50:95: {best_mAP50_95:.4f}\n")
        f.write(f"\nTEMPS\n   Debut: {time_stats['start_datetime']}\n")
        f.write(f"   Fin:   {time_stats['end_datetime']}\n")
        f.write(f"   Total: {time_stats['total_time_formatted']}\n")
    print(f"   Rapport sauvegarde: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO26-seg - Toitures Cadastrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  simple    YOLO26-seg standard, hyperparamètres fixes
  attention YOLO26-seg + CBAM sur les sorties neck (avant tête Detect)
  optimize  YOLO26-seg standard + recherche bayésienne des hyperparamètres (Optuna)
        """
    )
    parser.add_argument("--mode", choices=["simple", "attention", "optimize"],
                        default="simple", help="Mode d'entraînement (défaut: simple)")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_CONFIG["n_trials"])
    parser.add_argument("--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"])
    parser.add_argument("--cbam-reduction", type=int, default=CONFIG["cbam_reduction"])
    parser.add_argument("--cbam-kernel-size", type=int, default=CONFIG["cbam_kernel_size"],
                        choices=[3, 5, 7])
    args = parser.parse_args()

    OPTUNA_CONFIG["n_trials"]           = args.n_trials
    OPTUNA_CONFIG["n_epochs_per_trial"] = args.n_epochs_trial

    print("=" * 70)
    print("   YOLO26-seg - Segmentation des Toitures Cadastrales")
    print(f"   Mode: {args.mode.upper()}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"   GPU:  {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    aug_coeffs = load_aug_coefficients(CONFIG["classes"])
    yaml_path, _ = prepare_yolo_dataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        CONFIG["output_dir"], CONFIG["classes"], CONFIG["train_split"],
        aug_coeffs=aug_coeffs,
    )

    num_classes = len(CONFIG["classes"])

    trainer_cls = None

    if args.mode == "simple":
        print(f"\nArchitecture: YOLO26{CONFIG['model_size']}-seg (standard)")
        model        = get_model_simple()
        model_config = {"mode": "simple"}

    elif args.mode == "attention":
        cbam_r = args.cbam_reduction
        cbam_k = args.cbam_kernel_size
        print(f"\nArchitecture: YOLO26{CONFIG['model_size']}-seg + CBAM")
        print(f"   cbam_reduction={cbam_r}, cbam_kernel_size={cbam_k}")
        model        = get_model_attention(cbam_r, cbam_k)
        model_config = {"mode": "attention", "cbam_reduction": cbam_r, "cbam_kernel_size": cbam_k}
        trainer_cls  = _make_cbam_trainer(cbam_r, cbam_k)

    else:  # optimize
        print(f"\nArchitecture: YOLO26{CONFIG['model_size']}-seg (standard)")
        print("Lancement de l'optimisation bayesienne des hyperparametres...")
        best_params = _run_optimization(yaml_path)
        for key in ("learning_rate", "weight_decay", "momentum"):
            if key in best_params:
                CONFIG[key] = best_params[key]
        print(f"\nHyperparametres optimises appliques a l'entrainement.")
        model        = get_model_simple()
        model_config = {"mode": "optimize", "best_params": best_params}

    print(f"   Classes: {CONFIG['classes']}")
    gc.collect()

    _run_training(model, yaml_path, model_config, trainer_cls=trainer_cls)


if __name__ == "__main__":
    main()
