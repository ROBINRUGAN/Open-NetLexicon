#!/bin/bash
# Usage:
#   bash run_pretrain.sh
#   GPU=1 bash run_pretrain.sh
#   EPOCHS=30 bash run_pretrain.sh
#   SPLITS_DIR=splits/pretrain_all bash run_pretrain.sh
#   RUN_NAME=exp1 bash run_pretrain.sh

set -e
cd "$(dirname "$0")"

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-2e-4}
WARMUP=${WARMUP:-1000}
SPLITS_DIR=${SPLITS_DIR:-splits/pretrain_all}
RUN_NAME=${RUN_NAME:-""}


D_MODEL=${D_MODEL:-192}
N_LAYERS=${N_LAYERS:-4}
N_HEADS=${N_HEADS:-16}
D_FF=${D_FF:-768}
VQ_DIM=${VQ_DIM:-192}
VQ_CODEBOOK_SIZE=${VQ_CODEBOOK_SIZE:-128}
CONTRASTIVE_PROJ_DIM=${CONTRASTIVE_PROJ_DIM:-64}

echo "============================================"
echo " NetLexicon Pretrain"
echo "============================================"
echo " GPU:              ${GPU}"
echo " Splits:           ${SPLITS_DIR}"
echo " Epochs:           ${EPOCHS}"
echo " Batch Size:       ${BATCH_SIZE}"
echo " LR:               ${LR}"
echo " Warmup Steps:     ${WARMUP}"
echo " d_model:          ${D_MODEL}"
echo " n_layers:         ${N_LAYERS}"
echo " n_heads:          ${N_HEADS}"
echo " d_ff:             ${D_FF}"
echo " vq_dim:           ${VQ_DIM}"
echo " vq_codebook_size: ${VQ_CODEBOOK_SIZE}"
echo " contrastive_proj: ${CONTRASTIVE_PROJ_DIM}"
if [ -n "${RUN_NAME}" ]; then
    echo " run_name:         ${RUN_NAME}"
fi
echo "============================================"

CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${GPU} python pretrain/train.py \
    --splits_dir ${SPLITS_DIR} \
    --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --warmup_steps ${WARMUP} \
    --d_model ${D_MODEL} --n_layers ${N_LAYERS} --n_heads ${N_HEADS} --d_ff ${D_FF} \
    --vq_dim ${VQ_DIM} --vq_codebook_size ${VQ_CODEBOOK_SIZE} \
    --contrastive_proj_dim ${CONTRASTIVE_PROJ_DIM} \
    --device cuda:0"

if [ -n "${RUN_NAME}" ]; then
    CMD="${CMD} --run_name ${RUN_NAME}"
fi

eval ${CMD}
