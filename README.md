# NetLexicon

NetLexicon pretraining + fine-tuning code for encrypted traffic classification.

## Data (one-line requirement)

Prepare **per-flow PCAPs split by 5-tuple** (src/dst IP, src/dst port, protocol) for each dataset before running the pipeline.

## Pretrain

From `NetLexicon/`:

```bash
# 1) Build JSON features (writes to dataset/)
python3 prepare/build_dataset.py --workers 64

# 2) Compute normalization stats (writes norm_stats.json)
python3 prepare/compute_norm_stats.py

# 3) Prepare splits (writes splits.json + label_to_idx.json)
python3 prepare/prepare_splits.py

# 4) Pretrain
bash run_pretrain.sh
```

Common overrides:

```bash
GPU=1 EPOCHS=30 bash run_pretrain.sh
SPLITS_DIR=splits/pretrain_all GPU=0 bash run_pretrain.sh
RUN_NAME=exp1 GPU=0 bash run_pretrain.sh
```

## Fine-tune (single dataset)

Prepare dataset-specific splits first (example: `CipherSpectrum`):

```bash
python3 prepare/prepare_splits.py --dataset CipherSpectrum
```

Then fine-tune:

```bash
bash run_finetune.sh CipherSpectrum

# overrides
GPU=1 EPOCHS=20 PRETRAINED=checkpoints/pretrain/best.pt bash run_finetune.sh CipherSpectrum
```

## Ablations

All ablations are supported via flags in `finetune/finetune.py` and `pretrain/train.py`.

### no_stp

Disable next token-type prediction (STP) during pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 python3 pretrain/train.py --type_loss_weight 0
```

### no_sfa + no_vq

Disable stats-feature alignment (SFA) and VQ:

```bash
# pretrain: disable contrastive flow loss (SFA)
CUDA_VISIBLE_DEVICES=0 python3 pretrain/train.py --flow_loss_weight 0

# finetune: no VQ (use encoder hidden at [WIN])
CUDA_VISIBLE_DEVICES=0 python3 finetune/finetune.py \
  --pretrained checkpoints/pretrain/best.pt \
  --no_vq \
  --splits_dir splits/CipherSpectrum
```

### no_pretrained

Fine-tune from random init (no pretraining):

```bash
CUDA_VISIBLE_DEVICES=0 python3 finetune/finetune.py \
  --no_pretrained \
  --splits_dir splits/CipherSpectrum
```

## Codebook analysis

Analyze VQ code usage and its correlation with `window_stats` / labels on a chosen split.

```bash
CUDA_VISIBLE_DEVICES=0 python3 analysis/analyze_codebook.py \
  --ckpt checkpoints/finetune/CipherSpectrum/best.pt
  --splits_dir splits/CipherSpectrum \
  --split test \
  --full \
  --min_count 10
```

