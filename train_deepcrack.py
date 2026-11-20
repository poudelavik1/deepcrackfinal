#!/usr/bin/env python3
"""DeepCrack crack segmentation in one file: train, evaluate, predict.

    python train_deepcrack.py train    --data DATA --out runs/deepcrack_s0 [--seed 0]
    python train_deepcrack.py evaluate --checkpoint runs/deepcrack_s0/best.pt --data DATA --split test
    python train_deepcrack.py predict  --checkpoint runs/deepcrack_s0/best.pt --images photos/ --out predictions/
    python train_deepcrack.py summarize runs/new     # seeds in runs/new/seed_*: mean +- std, and the run to keep

Dataset (DATA), either layout works (image and mask files are paired by name; masks are crack > 0):
    DATA/train/images  DATA/train/masks  DATA/val/...  [DATA/test/...]       ready-made split, or
    DATA/images        DATA/masks                                            split 80/10/10 here, by source image

The training recipe is the result of the DeepCrack ablation in the thesis benchmark (13 Nov 2026 split;
pooled val IoU at the val-chosen threshold; reference = benchmark recipe, 0.5520 +- 0.0021 over seeds 0 1 2):

  model         DeepCrack (Liu et al. 2019): VGG-16-BN encoder (ImageNet weights), 5 side outputs fused by 1x1 conv
  optimizer     RAdam, lr 1e-4, weight decay 1e-4         final, seeds 0 1 2: 0.5540 +- 0.0027 (+0.0020, better on every
                                                            seed; Wilcoxon per tile p = 0.003). AdamW 1e-4 is as good
                                                            (0.5540): the gain is the lower lr. Adam-family at 1e-3: -0.05
  schedule      2 warm-up epochs, cosine to 1 % of lr, 80 epochs, early stop after 20 epochs without gain (not before 30)
  batch         16 random 256 x 256 crops, with a crack in the crop when the tile has one
                                                            uniform random crops instead: -0.0285 (largest effect found)
  augmentation  flips, 90-degree rotations, mild affine, brightness/contrast, HSV jitter (p 0.3), light blur/noise
                                                            no 90-degree rotations: -0.0072; no augmentation: -0.0069
                                                            no HSV alone: +0.0019 over 3 seeds, but not on top of lr 1e-4
                (14 added augmentations, e.g. CLAHE, elastic, CutMix, motion blur: all within seed noise)
  loss          BCE + soft Dice on the fused output
  stability     EMA of the weights (0.999, validated and saved), mixed precision, gradient clipping 1.0
  threshold     chosen on val (best pooled F1), stored in best.pt, used for test and predictions

`--recipe benchmark` gives the reference recipe (AdamW 3e-4). Ablation test set (once): final 0.5455 +- 0.0098 vs
benchmark 0.5360 +- 0.0085 IoU (+0.0095, every seed). Final model on the 16 Nov 2026 split, seeds 0 1 2: test IoU
0.5733 +- 0.0013, F1 0.7288 (results/ has every table, the recipe as JSON and the per-seed metrics).

Outputs of `train` in --out: best.pt (EMA weights + threshold), last.pt (resume), history.csv, recipe.json,
split.json (when the split was made here), val_metrics.json, test_metrics.json (if a test split exists), FINISHED.
An unfinished run resumes from last.pt; a finished one is skipped (--force retrains it).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

cv2.setNumThreads(0)   # the DataLoader workers already parallelize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
TILE_PATTERN = re.compile(r"^(?P<source>.*?)[_\-]r(?:ow)?[_\-]?\d+[_\-]c(?:ol)?[_\-]?\d+$", re.IGNORECASE)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LEVELS = 256   # probabilities are scored as 8-bit values; threshold k means prob >= k/255


# =========================================================================== #
# Recipe
# =========================================================================== #
@dataclass(frozen=True)
class Recipe:
    name: str = "tuned"
    epochs: int = 80
    batch_size: int = 16
    crop_size: int = 256
    optimizer: str = "radam"          # radam | adamw | adam | sgd
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    patience: int = 20
    min_epochs: int = 30
    amp: bool = True
    hsv_p: float = 0.3                # hue/saturation/value jitter, as in the benchmark (0 helps alone, not with lr 1e-4)
    eval_max_side: int = 1024         # larger images are predicted with a sliding window
    eval_window: int = 512
    eval_stride: int = 448
    tolerance_px: int = 2             # tolerance F1: a pixel counts within this distance of the other mask


RECIPES = {
    "tuned": Recipe(),
    "benchmark": Recipe(name="benchmark", optimizer="adamw", lr=3e-4),
}


# =========================================================================== #
# Data
# =========================================================================== #
Pair = tuple[str, Path, Path]   # (name, image path, mask path)


def normalized_stem(path: Path) -> str:
    """Lower-case stem without role suffixes (_mask, _label, _image, _img, _gt), so image and mask pair up."""
    stem, previous = path.stem.lower().strip(), None
    while stem != previous:
        previous = stem
        stem = re.sub(r"(?:[_\- ](?:mask|label|image|img|gt))$", "", stem)
    return stem


def source_of(name: str) -> str:
    """Source photo of a tile named <source>_r####_c####; any other file is its own source."""
    match = TILE_PATTERN.match(name)
    return (match.group("source") or name) if match else name


def pairs_in(folder: Path) -> list[Pair]:
    image_dir, mask_dir = folder / "images", folder / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected {folder}/images and {folder}/masks")

    def index(directory: Path) -> dict[str, Path]:
        return {normalized_stem(p): p for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    images, masks = index(image_dir), index(mask_dir)
    unpaired = sorted(set(images) ^ set(masks))
    if unpaired:
        raise SystemExit(f"{len(unpaired)} files without a partner in {folder}, e.g. {unpaired[:5]}")
    if not images:
        raise SystemExit(f"No images in {image_dir}")
    return [(key, images[key], masks[key]) for key in sorted(images)]


def split_folder(root: Path, split: str) -> Path | None:
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in SPLIT_NAMES[split]:
            return child
    return None


def load_splits(root: Path, out: Path | None = None, split_seed: int = 0) -> dict[str, list[Pair]]:
    """train / val / test pairs. A ready-made split is used as is; otherwise root/images + root/masks are split
    80/10/10 by source photo (tiles of one photo never end up in two splits), and the split is saved."""
    folders = {split: split_folder(root, split) for split in SPLIT_NAMES}
    if folders["train"] and folders["val"]:
        return {split: pairs_in(folder) for split, folder in folders.items() if folder}
    saved = out / "split.json" if out else None
    pairs = pairs_in(root)
    by_name = {name: (name, image, mask) for name, image, mask in pairs}
    if saved and saved.exists():
        names = json.loads(saved.read_text(encoding="utf-8"))
        return {split: [by_name[n] for n in names[split] if n in by_name] for split in ("train", "val", "test")}
    groups: dict[str, list[str]] = defaultdict(list)
    for name in by_name:
        groups[source_of(name)].append(name)
    if len(groups) < 3:
        raise SystemExit(f"Only {len(groups)} source photo(s): too few to split by photo. "
                         "Provide train/ val/ (test/) folders instead.")
    order = sorted(groups)
    random.Random(split_seed).shuffle(order)
    total, names = len(by_name), {"train": [], "val": [], "test": []}
    for group in order:   # whole photos: test and val each take ~10 % of the tiles, train the rest
        split = ("test" if len(names["test"]) < 0.1 * total else
                 "val" if len(names["val"]) < 0.1 * total else "train")
        names[split] += sorted(groups[group])
    if saved:
        saved.write_text(json.dumps(names, indent=1), encoding="utf-8")
    print(f"split made here (by source photo, seed {split_seed}): " + ", ".join(f"{k} {len(v)}" for k, v in names.items()))
    return {split: [by_name[n] for n in names[split]] for split in names if names[split]}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and {0, 1} uint8 mask of the same size."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read {mask_path}")
    image = read_image(image_path)
    if image.shape[:2] != mask.shape:   # a phone photo may carry an EXIF rotation that its mask did not get
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != mask.shape:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    threshold = 1 if mask.max() <= 1 else 128   # 0/1 masks and JPEG-compressed 0/255 masks
    return image, (mask >= threshold).astype(np.uint8)


def train_transform(recipe: Recipe, seed: int):
    import albumentations as A

    crop = recipe.crop_size
    steps: list[Any] = [
        A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=cv2.BORDER_REFLECT_101),
        A.CropNonEmptyMaskIfExists(height=crop, width=crop),   # most of a photo is background
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ]
    if recipe.hsv_p > 0:
        steps.append(A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=recipe.hsv_p))
    steps.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 3)), A.GaussNoise()], p=0.15))
    try:
        return A.Compose(steps, seed=seed)
    except TypeError:   # albumentations < 1.4.22 has no seed argument
        return A.Compose(steps)


def to_tensors(image: np.ndarray, mask: np.ndarray | None = None):
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    if mask is None:
        return image_tensor
    return image_tensor, torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()


class TrainSet(Dataset):
    def __init__(self, pairs: list[Pair], transform):
        self.pairs, self.transform = pairs, transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        _, image_path, mask_path = self.pairs[index]
        image, mask = read_pair(image_path, mask_path)
        out = self.transform(image=image, mask=mask)
        return to_tensors(out["image"], out["mask"])


class EvalSet(Dataset):
    """Whole tiles at native size (no augmentation)."""

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def sizes(self) -> list[tuple[int, int]]:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        sizes = []
        for _, _, mask_path in self.pairs:
            with Image.open(mask_path) as image:
                sizes.append((image.size[1], image.size[0]))
        return sizes

    def __getitem__(self, index: int):
        name, image_path, mask_path = self.pairs[index]
        image, mask = to_tensors(*read_pair(image_path, mask_path))
        return {"image": image, "mask": mask, "name": name}


class SizeBuckets(Sampler):
    """Batches of equally sized tiles; images that need a sliding window go one at a time."""

    def __init__(self, sizes: list[tuple[int, int]], batch_size: int, max_side: int):
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, size in enumerate(sizes):
            buckets[size].append(index)
        self.batches = []
        for size, indices in sorted(buckets.items()):
            step = 1 if max(size) > max_side else batch_size
            self.batches += [indices[i:i + step] for i in range(0, len(indices), step)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate_eval(batch):
    return {"image": torch.stack([b["image"] for b in batch]), "mask": torch.stack([b["mask"] for b in batch]),
            "name": [b["name"] for b in batch]}


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def default_workers() -> int:
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return 0 if os.name == "nt" else min(8, os.cpu_count() or 1)


def eval_loader(pairs: list[Pair], recipe: Recipe, workers: int, batch_size: int = 16) -> DataLoader:
    dataset = EvalSet(pairs)
    return DataLoader(dataset, batch_sampler=SizeBuckets(dataset.sizes(), batch_size, recipe.eval_max_side),
                      num_workers=workers, collate_fn=collate_eval, pin_memory=torch.cuda.is_available())


# =========================================================================== #
# Model: DeepCrack (Liu et al., Neurocomputing 2019)
# =========================================================================== #
class DeepCrack(nn.Module):
    """VGG-16-BN without the FC layers and the 5th pool; a 1x1 side output per stage, upsampled to the input
    size, and a 1x1 fusion of the 5 side outputs. Only the fused output is trained and predicted."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision

        features = torchvision.models.vgg16_bn(weights="DEFAULT" if pretrained else None).features
        self.stage1, self.stage2, self.stage3 = features[0:6], features[6:13], features[13:23]
        self.stage4, self.stage5 = features[23:33], features[33:43]
        self.side = nn.ModuleList(nn.Conv2d(channels, 1, 1) for channels in (64, 128, 256, 512, 512))
        self.fuse = nn.Conv2d(5, 1, 1)
        nn.init.constant_(self.fuse.weight, 0.2)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size, sides = x.shape[-2:], []
        for stage, side in zip((self.stage1, self.stage2, self.stage3, self.stage4, self.stage5), self.side):
            x = stage(x)
            sides.append(F.interpolate(side(x), size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(sides, 1))


class Normalized(nn.Module):
    """Takes RGB in [0, 1] and applies the ImageNet normalization itself (same parameter names as the
    benchmark code, so its DeepCrack checkpoints load here too)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def build_model(pretrained: bool) -> nn.Module:
    return Normalized(DeepCrack(pretrained))


class EMA:
    """Exponential moving average of the weights and BN statistics; it is what gets validated and saved."""

    def __init__(self, model: nn.Module, decay: float, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    probability = torch.sigmoid(logits)
    dice = (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice)


def make_optimizer(model: nn.Module, recipe: Recipe) -> torch.optim.Optimizer:
    params, name = model.parameters(), recipe.optimizer.lower()
    if name == "radam":
        return torch.optim.RAdam(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=recipe.lr, momentum=0.9, nesterov=True, weight_decay=recipe.weight_decay)
    raise SystemExit(f"Unknown optimizer {recipe.optimizer} (radam | adamw | adam | sgd)")


def make_scheduler(optimizer: torch.optim.Optimizer, recipe: Recipe):
    """Linear warm-up, then cosine decay to lr * min_lr_ratio; stepped once per epoch."""
    def factor(epoch: int) -> float:
        if recipe.warmup_epochs > 0 and epoch < recipe.warmup_epochs:
            return (epoch + 1) / (recipe.warmup_epochs + 1)
        progress = (epoch - recipe.warmup_epochs) / max(1.0, recipe.epochs - recipe.warmup_epochs)
        return recipe.min_lr_ratio + (1 - recipe.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


# =========================================================================== #
# Prediction (whole image, or a sliding window for large images)
# =========================================================================== #
def _forward_padded(model: nn.Module, images: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    height, width = images.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        images = F.pad(images, (0, pad_w, 0, pad_h), mode=mode)
    return torch.sigmoid(model(images)[..., :height, :width].float())


def _starts(length: int, window: int, stride: int) -> list[int]:
    return [0] if length <= window else list(range(0, length - window, stride)) + [length - window]


@torch.no_grad()
def predict_probability(model: nn.Module, images: torch.Tensor, recipe: Recipe, amp: bool) -> torch.Tensor:
    """Crack probability (N, 1, H, W); sliding window (overlaps averaged) above recipe.eval_max_side."""
    height, width = images.shape[-2:]
    with autocast(images.device, amp):
        if max(height, width) <= recipe.eval_max_side:
            return _forward_padded(model, images)
        window, stride = recipe.eval_window, recipe.eval_stride
        total = torch.zeros((images.shape[0], 1, height, width), device=images.device)
        count = torch.zeros((1, 1, height, width), device=images.device)
        boxes = [(t, l) for t in _starts(height, window, stride) for l in _starts(width, window, stride)]
        for start in range(0, len(boxes), 8):
            chunk = boxes[start:start + 8]
            crops = torch.cat([images[..., t:t + window, l:l + window] for t, l in chunk], 0)
            probability = _forward_padded(model, crops)
            for index, (t, l) in enumerate(chunk):
                total[..., t:t + window, l:l + window] += probability[index * images.shape[0]:(index + 1) * images.shape[0]]
                count[..., t:t + window, l:l + window] += 1
        return total / count


# =========================================================================== #
# Metrics (pooled over all pixels of a split, as in the benchmark)
# =========================================================================== #
def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def scores(tp: float, fp: float, fn: float) -> dict[str, float]:
    return {"iou": ratio(tp, tp + fp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel (inf when there is none)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def score_split(tiles: list[tuple[np.ndarray, np.ndarray]], threshold_index: int | None,
                tolerance: int) -> dict[str, Any]:
    """Metrics of (8-bit probability, {0,1} mask) tiles. threshold_index None: choose the best pooled F1 here."""
    positive, negative = np.zeros(LEVELS, np.int64), np.zeros(LEVELS, np.int64)
    for prob_u8, gt in tiles:
        gt = gt.astype(bool)
        positive += np.bincount(prob_u8[gt], minlength=LEVELS)
        negative += np.bincount(prob_u8[~gt], minlength=LEVELS)
    tp_curve = np.cumsum(positive[::-1])[::-1][1:].astype(np.float64)     # prob >= k/255, k = 1..255
    fp_curve = np.cumsum(negative[::-1])[::-1][1:].astype(np.float64)
    fn_curve = positive.sum() - tp_curve
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_curve = np.where(2 * tp_curve + fp_curve + fn_curve > 0, 2 * tp_curve / (2 * tp_curve + fp_curve + fn_curve), 0)
        precision_curve = np.where(tp_curve + fp_curve > 0, tp_curve / (tp_curve + fp_curve), 1.0)
        recall_curve = np.where(tp_curve + fn_curve > 0, tp_curve / (tp_curve + fn_curve), 0.0)
    recall_ext, precision_ext = np.concatenate([recall_curve, [0.0]]), np.concatenate([precision_curve, [1.0]])
    average_precision = float(np.sum((recall_ext[:-1] - recall_ext[1:]) * precision_ext[:-1]))
    chosen_here = threshold_index is None
    if chosen_here:
        threshold_index = int(np.argmax(f1_curve)) + 1
    k = threshold_index - 1
    result = scores(tp_curve[k], fp_curve[k], fn_curve[k])
    n_pred = n_gt = pred_hit = gt_hit = 0
    for prob_u8, gt in tiles:
        pred, gt = prob_u8 >= threshold_index, gt.astype(bool)
        n_pred += int(pred.sum())
        n_gt += int(gt.sum())
        pred_hit += int((distance_to(gt)[pred] <= tolerance).sum())
        gt_hit += int((distance_to(pred)[gt] <= tolerance).sum())
    tol_precision, tol_recall = ratio(pred_hit, n_pred), ratio(gt_hit, n_gt)
    result.update({f"tol_f1_{tolerance}px": ratio(2 * tol_precision * tol_recall, tol_precision + tol_recall),
                   f"tol_precision_{tolerance}px": tol_precision, f"tol_recall_{tolerance}px": tol_recall,
                   "average_precision": average_precision})
    return {"threshold": threshold_index / 255.0, "threshold_index": threshold_index,
            "threshold_source": "best pooled F1 on this split" if chosen_here else "chosen on val",
            "n_tiles": len(tiles), "metrics": result}


@torch.no_grad()
def collect(model: nn.Module, loader: DataLoader, recipe: Recipe, device: torch.device) -> list[tuple[np.ndarray, np.ndarray]]:
    model.eval()
    tiles = []
    for batch in loader:
        probability = predict_probability(model, batch["image"].to(device, non_blocking=True), recipe, recipe.amp)
        prob_u8 = (probability * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        masks = batch["mask"].to(torch.uint8).numpy()
        tiles += [(prob_u8[i, 0], masks[i, 0]) for i in range(len(batch["name"]))]
    return tiles


@torch.no_grad()
def quick_val_iou(model: nn.Module, loader: DataLoader, recipe: Recipe, device: torch.device) -> dict[str, float]:
    """Pooled val IoU / F1 at threshold 0.5, for early stopping."""
    model.eval()
    tp = fp = fn = 0.0
    for batch in loader:
        pred = predict_probability(model, batch["image"].to(device, non_blocking=True), recipe, recipe.amp) >= 0.5
        truth = batch["mask"].to(device) >= 0.5
        tp += float((pred & truth).sum())
        fp += float((pred & ~truth).sum())
        fn += float((~pred & truth).sum())
    return scores(tp, fp, fn)


# =========================================================================== #
# Commands
# =========================================================================== #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def atomic_save(obj: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def recipe_from_args(args: argparse.Namespace) -> Recipe:
    recipe = RECIPES[args.recipe]
    changes = {key: getattr(args, key) for key in ("epochs", "batch_size", "lr", "optimizer", "crop_size")
               if getattr(args, key, None) is not None}
    if getattr(args, "no_amp", False):
        changes["amp"] = False
    return replace(recipe, **changes)


def train(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FINISHED").exists() and not args.force:
        print(f"[skip] {out} is finished (--force retrains it)")
        return 0
    recipe = recipe_from_args(args)
    last_path, best_path, history_path = out / "last.pt", out / "best.pt", out / "history.csv"
    resume = last_path.exists() and not args.force
    if not resume:
        for stale in (last_path, best_path, history_path, out / "FINISHED", out / "split.json"):
            stale.unlink(missing_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = args.workers if args.workers is not None else default_workers()
    splits = load_splits(Path(args.data), out)
    if args.max_tiles:   # quick test runs only
        splits = {k: v[:args.max_tiles] for k, v in splits.items()}
    (out / "recipe.json").write_text(json.dumps({**asdict(recipe), "seed": args.seed, "data": str(args.data)}, indent=2),
                                     encoding="utf-8")

    model = build_model(not args.no_pretrained).to(device)
    optimizer = make_optimizer(model, recipe)
    scheduler = make_scheduler(optimizer, recipe)
    scaler = grad_scaler(device, recipe.amp)
    ema = EMA(model, recipe.ema_decay)
    state = {"epoch": 0, "best_iou": -1.0, "best_epoch": 0, "bad_epochs": 0}
    if resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.module.load_state_dict(checkpoint["ema"])
        ema.updates = checkpoint["ema_updates"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        torch.set_rng_state(checkpoint["torch_rng"])
        print(f"[resume] from epoch {state['epoch']} (best val IoU {state['best_iou']:.4f} at epoch {state['best_epoch']})")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_set = TrainSet(splits["train"], train_transform(recipe, args.seed))
    loader = DataLoader(train_set, batch_size=recipe.batch_size, shuffle=True, drop_last=len(train_set) > recipe.batch_size,
                        num_workers=workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=workers > 0)
    val_loader = eval_loader(splits["val"], recipe, workers)
    print(f"[train] DeepCrack | recipe {recipe.name}: {recipe.optimizer} lr {recipe.lr:g}, HSV p {recipe.hsv_p} | "
          f"seed {args.seed} | {device_name(device)} | workers {workers}\n"
          f"        tiles: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits.get('test', []))} | out {out}",
          flush=True)

    while state["epoch"] < recipe.epochs:
        epoch, started = state["epoch"], time.time()
        model.train()
        total_loss, batches, skipped = 0.0, 0, 0
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with autocast(device, recipe.amp):
                logits = model(images)
            loss = bce_dice(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):   # one fp16 overflow should not end the run
                skipped += 1
                if skipped > max(20, len(loader) // 50):
                    raise FloatingPointError(f"non-finite loss in {skipped} steps of epoch {epoch + 1}: diverged")
                continue
            scaler.scale(loss).backward()
            if recipe.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += float(loss.detach())
            batches += 1
        val = quick_val_iou(ema.module, val_loader, recipe, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        state["epoch"] = epoch + 1
        improved = val["iou"] > state["best_iou"]
        if improved:
            state.update(best_iou=val["iou"], best_epoch=epoch + 1, bad_epochs=0)
            atomic_save({"model": ema.module.state_dict(), "recipe": asdict(recipe), "epoch": epoch + 1,
                         "val_iou_at_0.5": val["iou"], "threshold": None, "arch": "deepcrack_vgg16bn"}, best_path)
        else:
            state["bad_epochs"] += 1
        seconds = time.time() - started
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["epoch", "lr", "train_loss", "val_iou", "val_f1", "val_precision", "val_recall", "seconds"])
            writer.writerow([epoch + 1, f"{lr:.3e}", round(total_loss / max(1, batches), 5), round(val["iou"], 5),
                             round(val["f1"], 5), round(val["precision"], 5), round(val["recall"], 5), round(seconds, 1)])
        atomic_save({"model": model.state_dict(), "ema": ema.module.state_dict(), "ema_updates": ema.updates,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "state": state, "torch_rng": torch.get_rng_state()}, last_path)
        print(f"epoch {epoch + 1:3d}/{recipe.epochs} | lr {lr:.2e} | loss {total_loss / max(1, batches):.4f} | "
              f"val IoU {val['iou']:.4f} F1 {val['f1']:.4f}{' *' if improved else ''} | {seconds:.0f}s", flush=True)
        if state["bad_epochs"] >= recipe.patience and state["epoch"] >= recipe.min_epochs:
            print(f"[early stop] no val IoU gain for {recipe.patience} epochs")
            break

    # Threshold on val (best pooled F1) with the best checkpoint; then the test split once, with that threshold.
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    summary = {"best_epoch": state["best_epoch"], "epochs_run": state["epoch"], "seed": args.seed,
               "recipe": asdict(recipe), "device": device_name(device)}
    val_result = score_split(collect(model, val_loader, recipe, device), None, recipe.tolerance_px)
    (out / "val_metrics.json").write_text(json.dumps({**summary, **val_result}, indent=2), encoding="utf-8")
    best["threshold"] = val_result["threshold"]
    atomic_save(best, best_path)
    print(f"[val]  threshold {val_result['threshold']:.3f} | " + metrics_line(val_result["metrics"]), flush=True)
    if splits.get("test"):
        test_loader = eval_loader(splits["test"], recipe, workers)
        test_result = score_split(collect(model, test_loader, recipe, device), val_result["threshold_index"],
                                  recipe.tolerance_px)
        (out / "test_metrics.json").write_text(json.dumps({**summary, **test_result}, indent=2), encoding="utf-8")
        print(f"[test] threshold {test_result['threshold']:.3f} | " + metrics_line(test_result["metrics"]), flush=True)
    (out / "FINISHED").write_text(json.dumps({**summary, "best_val_iou_at_0.5": state["best_iou"]}, indent=2),
                                  encoding="utf-8")
    print(f"[done] best epoch {state['best_epoch']} | {out / 'best.pt'}")
    return 0


def metrics_line(metrics: dict[str, float]) -> str:
    tolerance = next(key for key in metrics if key.startswith("tol_f1_"))
    return (f"IoU {metrics['iou']:.4f}  F1 {metrics['f1']:.4f}  precision {metrics['precision']:.4f}  "
            f"recall {metrics['recall']:.4f}  {tolerance.replace('tol_f1_', 'F1@')} {metrics[tolerance]:.4f}  "
            f"AP {metrics['average_precision']:.4f}")


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, Recipe, float | None]:
    """best.pt of this script, or of the benchmark code (deepcrack runs); returns (model, recipe, threshold)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    fields = Recipe.__dataclass_fields__
    recipe = Recipe(**{k: v for k, v in checkpoint.get("recipe", {}).items() if k in fields})
    if "config" in checkpoint and "recipe" not in checkpoint:   # a benchmark checkpoint: take its evaluation settings
        data = checkpoint["config"].get("data", {})
        recipe = replace(recipe, eval_max_side=int(data.get("eval_max_side", 1024)),
                         eval_window=int(data.get("eval_window", 512)), eval_stride=int(data.get("eval_stride", 448)),
                         amp=bool(checkpoint["config"].get("train", {}).get("amp", True)))
    return model, recipe, checkpoint.get("threshold")


def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, recipe, threshold = load_checkpoint(args.checkpoint, device)
    if args.threshold is not None:
        threshold = args.threshold
    splits = load_splits(Path(args.data), Path(args.checkpoint).parent)
    if args.split not in splits:
        raise SystemExit(f"No {args.split} split in {args.data}")
    if threshold is None and args.split != "val":
        raise SystemExit("The checkpoint has no threshold: pass --threshold (chosen on val), or evaluate --split val.")
    loader = eval_loader(splits[args.split], recipe, args.workers if args.workers is not None else default_workers())
    index = None if threshold is None else max(1, min(255, round(threshold * 255)))
    result = score_split(collect(model, loader, recipe, device), index, recipe.tolerance_px)
    print(f"[{args.split}] threshold {result['threshold']:.3f} ({result['threshold_source']}) | "
          + metrics_line(result["metrics"]))
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def predict(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, recipe, threshold = load_checkpoint(args.checkpoint, device)
    if args.threshold is not None:
        threshold = args.threshold
    if threshold is None:
        threshold = 0.5
        print("WARNING: the checkpoint has no threshold chosen on val; using 0.5 (pass --threshold)")
    source, out = Path(args.images), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for path in files:
        image = read_image(path)
        probability = predict_probability(model, to_tensors(image).unsqueeze(0).to(device), recipe, recipe.amp)[0, 0]
        prob_u8 = (probability * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        mask = (prob_u8 >= max(1, round(threshold * 255))).astype(np.uint8) * 255
        cv2.imwrite(str(out / f"{path.stem}_prob.png"), prob_u8)
        cv2.imwrite(str(out / f"{path.stem}_mask.png"), mask)
        if args.overlay:
            overlay = image.copy()
            overlay[mask > 0] = (255, 0, 0)
            blended = cv2.addWeighted(image, 0.45, overlay, 0.55, 0)
            cv2.imwrite(str(out / f"{path.stem}_overlay.jpg"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"{path.name}: {int((mask > 0).sum())} crack pixels ({(mask > 0).mean() * 100:.2f} %)")
    print(f"[done] {len(files)} images -> {out} (threshold {threshold:.3f})")
    return 0


# =========================================================================== #
# Summary over seeds
# =========================================================================== #
def summarize(args: argparse.Namespace) -> int:
    """Per-seed metrics of the finished runs in a folder, their mean +- std, and the run to keep (best val IoU)."""
    root = Path(args.runs)
    runs = sorted(path.parent for path in root.glob("*/FINISHED") if (path.parent / "val_metrics.json").exists())
    if not runs:
        raise SystemExit(f"No finished runs (*/FINISHED with val_metrics.json) under {root}")
    rows = []
    for run in runs:
        row: dict[str, Any] = {"run": run.name}
        for split in ("val", "test"):
            path = run / f"{split}_metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            row.update(seed=result.get("seed"), best_epoch=result.get("best_epoch"), threshold=result["threshold"])
            row.update({f"{split}_{key}": value for key, value in result["metrics"].items()})
        rows.append(row)
    columns = [c for c in rows[0] if c.startswith(("val_", "test_"))]
    columns = [c for c in columns if all(c in row for row in rows)]
    shown = [c for c in columns if c.split("_", 1)[1] in ("iou", "f1", "precision", "recall", "average_precision")
             or c.split("_", 1)[1].startswith("tol_f1_")]

    def short(column: str) -> str:
        return column.replace("average_precision", "AP").replace("precision", "P").replace("recall", "R") \
            .replace("tol_f1_", "F1@")

    print(f"{'run':<12} {'seed':>4} {'epoch':>5} {'thr':>6} " + " ".join(f"{short(c):>10}" for c in shown))
    for row in rows:
        print(f"{row['run']:<12} {row.get('seed', ''):>4} {row.get('best_epoch', ''):>5} {row['threshold']:>6.3f} "
              + " ".join(f"{row[c]:>10.4f}" for c in shown))
    summary: dict[str, Any] = {"runs": rows, "n": len(rows), "mean": {}, "std": {}}
    for column in columns:
        values = np.array([row[column] for row in rows], dtype=np.float64)
        summary["mean"][column] = float(values.mean())
        summary["std"][column] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    print(f"{'mean':<12} {'':>4} {'':>5} {'':>6} " + " ".join(f"{summary['mean'][c]:>10.4f}" for c in shown))
    print(f"{'std':<12} {'':>4} {'':>5} {'':>6} " + " ".join(f"{summary['std'][c]:>10.4f}" for c in shown))
    keep = max(rows, key=lambda row: row["val_iou"])
    summary["keep"] = keep["run"]
    print(f"\nkeep: {root / keep['run'] / 'best.pt'} (best val IoU {keep['val_iou']:.4f}; chosen on val, not test)")
    if "test_iou" in summary["mean"] and len(rows) > 1:
        print(f"report: test IoU {summary['mean']['test_iou']:.4f} +- {summary['std']['test_iou']:.4f} over {len(rows)} seeds")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "seed", "best_epoch", "threshold"] + columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        for name in ("mean", "std"):
            writer.writerow({"run": name, **summary[name]})
    print(f"[done] {root / 'summary.json'}, {root / 'summary.csv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", help="train on DATA, choose the threshold on val, score test")
    p.add_argument("--data", required=True, help="dataset folder (see the top of this file)")
    p.add_argument("--out", required=True, help="run folder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recipe", choices=sorted(RECIPES), default="tuned")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer", choices=("radam", "adamw", "adam", "sgd"))
    p.add_argument("--crop-size", type=int)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="random init (no ImageNet download)")
    p.add_argument("--workers", type=int, help="DataLoader workers (default: SLURM_CPUS_PER_TASK, or 0 on Windows)")
    p.add_argument("--max-tiles", type=int, default=0, help="use only the first N tiles per split (quick tests)")
    p.add_argument("--force", action="store_true", help="retrain even if the run folder is finished")
    p.set_defaults(handler=train)

    p = commands.add_parser("evaluate", help="score a checkpoint on a split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--threshold", type=float, help="default: the one stored in the checkpoint")
    p.add_argument("--workers", type=int)
    p.add_argument("--save", help="write the metrics to this JSON file")
    p.set_defaults(handler=evaluate)

    p = commands.add_parser("predict", help="crack masks for new images (any size)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True, help="an image or a folder of images")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, help="default: the one stored in the checkpoint")
    p.add_argument("--overlay", action="store_true", help="also save the cracks drawn in red over the image")
    p.set_defaults(handler=predict)

    p = commands.add_parser("summarize", help="mean +- std over the seeds (run folders) in a folder, and the run to keep")
    p.add_argument("runs", help="folder with one run folder per seed, e.g. runs/new (seed_0, seed_1, seed_2)")
    p.set_defaults(handler=summarize)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
