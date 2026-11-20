# DeepCrack: final training code

One file, [`train_deepcrack.py`](train_deepcrack.py), trains DeepCrack for concrete crack
segmentation on any dataset of image/mask pairs, chooses the decision threshold on the
validation set, scores the test set once, and predicts crack masks for new photos of any size.

The training recipe is the outcome of the DeepCrack ablation in the thesis benchmark
(baseline-selection repository, `configs/ablation/deepcrack.yaml`): 9 optimizers on a
learning-rate grid and 24 augmentation changes, each compared with the benchmark recipe.

## Quick start

```bash
pip install -r requirements.txt
python train_deepcrack.py train --data /path/to/dataset --out runs/deepcrack_s0 --seed 0
python train_deepcrack.py predict --checkpoint runs/deepcrack_s0/best.pt --images new_photos/ --out predictions/ --overlay
```

`train` ends with the val metrics at the chosen threshold and, when the dataset has a test
split, the test metrics at that same threshold. `predict` writes `<photo>_prob.png` (probability
0-255), `<photo>_mask.png` (crack = 255) and, with `--overlay`, `<photo>_overlay.jpg`.

## Dataset

Image and mask files are paired by name (`tile_0001.jpg` with `tile_0001.png` or
`tile_0001_mask.png`). Masks may be 0/255 or 0/1; any value above half is crack. Either layout:

```
dataset/train/images  dataset/train/masks        a ready-made split (test/ is optional)
dataset/val/images    dataset/val/masks
dataset/test/images   dataset/test/masks

dataset/images        dataset/masks              no split yet: made here, 80/10/10 by source photo
```

When the split is made here, tiles named `<photo>_r0003_c0012` are grouped by `<photo>`, so the
tiles of one photo never end up in two splits (otherwise neighbouring tiles leak into the test
set). The split is saved as `split.json` in the run folder and reused on every rerun.

## The recipe, and the evidence for each choice

The ablation ran in three stages on the 13 Nov 2026 split (13.6k training tiles), always against
the benchmark recipe (reference: val IoU 0.5520 ± 0.0021 over seeds 0 1 2). Values are pooled val IoU
at the val-chosen threshold. Stage 1 screened 56 single-seed changes (a change counts only beyond
±0.003, about the seed spread); stage 3 retrained the best ones with seeds 1 and 2.

| Recipe, seeds 0 1 2 (stage 3) | Val IoU | Δ vs reference | Per seed |
|---|---|---|---|
| **RAdam lr 1e-4 (final recipe)** | **0.5540 ± 0.0027** | **+0.0020** | +0.0036, +0.0016, +0.0009; Wilcoxon per tile p = 0.003 |
| AdamW lr 1e-4 (control) | 0.5540 ± 0.0019 | +0.0020 | |
| AdamW 3e-4, no HSV jitter | 0.5539 ± 0.0024 | +0.0019 | |
| Reference: AdamW lr 3e-4 (benchmark) | 0.5520 ± 0.0021 | | |

What this means: the gain is small (+0.4 % relative) but consistent, and it comes from the **lower
learning rate**, not from RAdam: AdamW at 1e-4 is just as good. RAdam 1e-4 is kept because it was
the pre-registered rule (best three-seed mean); `--optimizer adamw` gives the same result.
Removing HSV jitter helps by the same amount on its own, but not on top of the lower lr
(RAdam 1e-4 + no HSV, seed 0: 0.5552, below both parts), so HSV jitter stays.

| Setting | Value | Evidence (stage 1, Δ val IoU, seed 0) |
|---|---|---|
| Model | DeepCrack: VGG-16-BN encoder (ImageNet weights), 5 side outputs, 1×1 fusion | The model of this ablation (the crack-specific baseline of the thesis benchmark) |
| Optimizer | **RAdam, lr 1e-4**, weight decay 1e-4 | Best of 30 optimizer/lr settings (+0.0047 with one seed, +0.0020 over three). RAdam at 3e-5: −0.0010; at 3e-4: −0.012; at 1e-3: −0.053 |
| Learning rate | 1e-4, 2 warm-up epochs, cosine to 1 % | The lr matters more than the optimizer: every Adam-type optimizer is best at 1e-4 (Adamax: 3e-4), 0.552–0.557, and collapses at 1e-3 (~0.50); 3e-5 is already too low |
| Crop | 256 × 256, containing a crack when the tile has one | Uniform random crops instead: **−0.0285**, the largest effect found |
| Geometric augmentation | flips, 90° rotations, mild affine (±15°, scale 0.9–1.1) | No 90° rotations: −0.0072 (overfits, stops at epoch 61). No augmentation at all: −0.0069 (stops at epoch 43) |
| Photometric augmentation | brightness/contrast ±0.15, HSV jitter (p 0.3), light blur/noise | Stronger brightness/contrast (±0.3): −0.0071. No HSV: see above |
| Added augmentations | none | Elastic, grid distortion, perspective, CLAHE, gamma, shadow, grayscale, sharpen, motion blur, downscale, sensor noise, JPEG, coarse dropout, CutMix: all within ±0.003 |
| Loss | BCE + soft Dice on the fused output | Benchmark protocol |
| Stability | EMA of weights (0.999), mixed precision, gradient clipping 1.0 | Benchmark protocol v2 (removed the ±0.3 epoch-to-epoch swings of v1) |
| Length | 80 epochs, early stop after 20 without gain, not before epoch 30 | Most runs used the full 80 epochs |
| Threshold | best pooled F1 on val, stored in `best.pt` | Used unchanged for test and predictions |

`--recipe benchmark` runs the reference recipe (AdamW 3e-4).

## Commands

```bash
python train_deepcrack.py train    --data DATA --out RUN [--seed 0] [--recipe tuned|benchmark]
                                   [--epochs 80] [--batch-size 16] [--lr 1e-4] [--optimizer radam] [--crop-size 256]
python train_deepcrack.py evaluate --checkpoint RUN/best.pt --data DATA --split test [--threshold T] [--save m.json]
python train_deepcrack.py predict  --checkpoint RUN/best.pt --images FOLDER_OR_IMAGE --out OUT [--overlay]
python train_deepcrack.py summarize RUNS        # RUNS/seed_0, seed_1, ...: mean ± std, and the run to keep
```

- A run folder that has `FINISHED` is skipped (`--force` retrains it); an interrupted run resumes
  from `last.pt`.
- Images larger than 1024 px are predicted with a 512 px sliding window (stride 448) and the
  overlaps are averaged, so full drone photos can be passed to `predict` directly.
- Seeds change the result by about ±0.002 val IoU. For numbers to report, train seeds 0, 1 and 2
  and give the mean ± std: `summarize` prints them per split, writes `summary.json` / `summary.csv`,
  and names the run to keep (best val IoU, so the choice never looks at test).

## Outputs of `train`

| File | Content |
|---|---|
| `best.pt` | EMA weights of the best epoch (val IoU at 0.5), the recipe and the val-chosen threshold |
| `last.pt` | Everything needed to resume |
| `history.csv` | Per epoch: lr, train loss, val IoU / F1 / precision / recall at 0.5, seconds |
| `val_metrics.json`, `test_metrics.json` | IoU, F1, precision, recall, F1 at 2 px tolerance, average precision, threshold |
| `recipe.json`, `split.json` | Settings of the run; the split, when it was made here |

The metrics are the benchmark's: pooled over all pixels of a split, F1 at 2 px tolerance counts a
predicted pixel as correct within 2 px of a labelled one and vice versa.

## On the ASL cluster

```bash
git clone https://github.com/poudelavik1/deepcrackfinal.git ~/deepcrackfinal && cd ~/deepcrackfinal
DATA=~/new_dataset OUT=~/deepcrack_runs/new_s0 SEED=0 sbatch slurm/train.sbatch    # one seed
DATA=~/new_dataset OUT=~/deepcrack_runs/new sbatch --array=0-2 slurm/train.sbatch    # seeds 0 1 2
```

With `--array`, task n trains seed n in `OUT/seed_n` and then runs `summarize OUT`, so after the
last seed `OUT/summary.csv` holds the mean ± std. On an account limited to one running job, the
seeds simply run one after another.

The job puts the conda environment on `PATH` (compute nodes have no conda) and reads the
VGG-16-BN ImageNet weights from `~/weights_cache/torch`, which the benchmark already filled. On a
new account, download them once on the login node:
`TORCH_HOME=~/weights_cache/torch python -c "import torchvision; torchvision.models.vgg16_bn(weights='DEFAULT')"`.
A full run (13.6k tiles of 256 px, 80 epochs) takes about 1.3 h on an RTX A6000; time grows with
the number of training tiles (about 1.9 h for 19.6k).

## Using a checkpoint from the benchmark

Checkpoints of the benchmark's DeepCrack runs load directly. They have no stored threshold, so
pass the one from that run's `eval/val_metrics.json`:

```bash
python train_deepcrack.py predict --checkpoint ~/crack_benchmark_runs_v2/ablation_deepcrack/opt_radam_lr1e-4/seed_0/best.pt \
    --threshold 0.584 --images photos/ --out predictions/ --overlay
```
