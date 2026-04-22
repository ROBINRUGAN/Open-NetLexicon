#!/bin/bash
# Fine-tune on a single dataset.
#
# Usage:
#   bash run_finetune.sh <DATASET>
#   bash run_finetune.sh --list
#   bash run_finetune.sh -h

set -e
cd "$(dirname "$0")"

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-20}
PRETRAINED=${PRETRAINED:-checkpoints/pretrain/best.pt}

# Model args (must match pretraining).
D_MODEL=${D_MODEL:-192}
N_LAYERS=${N_LAYERS:-4}
N_HEADS=${N_HEADS:-16}
D_FF=${D_FF:-768}
VQ_DIM=${VQ_DIM:-192}
VQ_CODEBOOK_SIZE=${VQ_CODEBOOK_SIZE:-128}
CONTRASTIVE_PROJ_DIM=${CONTRASTIVE_PROJ_DIM:-64}

MODEL_ARGS="--d_model ${D_MODEL} --n_layers ${N_LAYERS} --n_heads ${N_HEADS} --d_ff ${D_FF} \
  --vq_dim ${VQ_DIM} --vq_codebook_size ${VQ_CODEBOOK_SIZE} \
  --contrastive_proj_dim ${CONTRASTIVE_PROJ_DIM}"

DATASETS=(
  "CipherSpectrum"
  "CrossPlatform"
  "cstnet"
  "ISCXVPN2016_VPN"
  "USTC-TFC"
)

usage() {
    cat <<'EOF'
Usage:
  bash run_finetune.sh <DATASET>
  bash run_finetune.sh --list

Optional env vars:
  GPU=0|1|...                  GPU index (default: 0)
  EPOCHS=20                    epochs (default: 20)
  PRETRAINED=path/to/best.pt   pretrained checkpoint (default: checkpoints/pretrain/best.pt)
  D_MODEL=192 N_LAYERS=4 N_HEADS=16 D_FF=768 VQ_DIM=192 VQ_CODEBOOK_SIZE=128 CONTRASTIVE_PROJ_DIM=64

Examples:
  bash run_finetune.sh USTC-TFC
  GPU=1 EPOCHS=10 bash run_finetune.sh cstnet
EOF
}

list_datasets() {
    for ds in "${DATASETS[@]}"; do
        echo "${ds}"
    done
}

is_valid_dataset() {
    local target="$1"
    for ds in "${DATASETS[@]}"; do
        if [[ "${ds}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ $# -eq 0 ]]; then
    usage
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    --list)
        list_datasets
        exit 0
        ;;
esac

DS="$1"
if ! is_valid_dataset "${DS}"; then
    echo "Error: unknown dataset '${DS}'"
    echo ""
    echo "Available datasets:"
    list_datasets
    exit 2
fi

echo "[fine-tune] dataset=${DS} gpu=${GPU} epochs=${EPOCHS}"
echo "[fine-tune] pretrained=${PRETRAINED}"

CUDA_VISIBLE_DEVICES=${GPU} python finetune/finetune.py \
    --pretrained ${PRETRAINED} --no_vq \
    --splits_dir splits/${DS} --run_name ${DS} \
    --epochs ${EPOCHS} --device cuda:0 \
    ${MODEL_ARGS}

echo "[fine-tune] done: ${DS}"
echo ""
echo "Eval (example):"
echo "  mkdir -p evaluation/results/${DS} visualization/json"
echo "  CUDA_VISIBLE_DEVICES=${GPU} python evaluation/evaluate.py --mode finetune \\"
echo "    --checkpoint checkpoints/finetune/${DS}/best.pt \\"
echo "    --splits_dir splits/${DS} ${MODEL_ARGS} \\"
echo "    --metrics_json evaluation/results/${DS}/test_metrics.json \\"
echo "    --per_class_csv evaluation/results/${DS}/per_class.csv \\"
echo "    --save_per_class visualization/json/per_class_${DS}.json"
