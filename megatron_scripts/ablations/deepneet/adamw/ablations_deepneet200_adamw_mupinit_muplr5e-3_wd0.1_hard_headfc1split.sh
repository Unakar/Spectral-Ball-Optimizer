#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

export WANDB_MODE=offline

TOTAL_TOKENS=50000000000 # 大概50B！

WORLD_SIZE=$((8 * $PET_NNODES))
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-4}

GLOBAL_BATCH=1024
TRAIN_ITER=$((TOTAL_TOKENS / GLOBAL_BATCH / 4096))

REPO_PATH="results/optimizer_arena_v2/deepneet"
JOB_NAME=ablations_deepneet200_adamw_mupinit_muplr5e-3_wd0.1_hard_headfc1split
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"

LOG_DIR="${REPO_PATH}/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $WANDB_PATH

cd code/Megatron-LM
PRETRAIN_SCRIPT="code/Megatron-LM/pretrain_gpt.py"


BASE_PATH="${BASE_PATH:-data/merged_data}"
DATA_PATH=""

while IFS= read -r file; do
    common_prefix=${file%".bin"}
    DATA_PATH+="1 ${common_prefix} "
done < <(find "$BASE_PATH" -type f -path "**.bin")

DATA_PATH_CACHE="data/merged_data_cache"

DATA_ARGS=(
    --tokenizer-model models/OLMo-2-1124-7B
    --tokenizer-type HuggingFaceTokenizer
    --data-path $DATA_PATH
    --data-cache-path ${DATA_PATH_CACHE}
    --train-iters $TRAIN_ITER
    --split 99,1,0
    --num-dataset-builder-threads 128
    --num-workers 16
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
)

TRAINING_ARGS=(
    --lr 5e-3
    --lr-warmup-iters 500
    --lr-decay-style cosine
    --min-lr 5e-4
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --use-distributed-optimizer
    --weight-decay 0.1
    --clip-grad 1.0
    --recompute-activations
    --recompute-granularity full
)


MODEL_ARGS=(
    --num-layers 200
    --hidden-size 512
    --ffn-hidden-size 1536
    --group-query-attention
    --num-attention-heads 8
    --num-query-groups 4
    --norm-epsilon 1e-6
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 40960
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --qk-layernorm
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --init-method-std 0.02
    --split-qkv-init-mode head
    --spectral-mup-init
)


CKPT_ARGS=(
    --ckpt-format "torch_dist"
    --save-interval 250000
    --no-save-optim
    --async-save
    --save $CHECKPOINT_PATH
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --micro-batch-size 8
    --global-batch-size ${GLOBAL_BATCH}
)

LOGGER_ARGS=(
    --log-throughput
    --log-interval 10
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-dir ${TENSORBOARD_PATH})

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes $PET_NNODES
    --node_rank $PET_NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

WANDB_ARGS=(
    --wandb-project optimizer_arena_v2
    --wandb-exp-name $JOB_NAME
    --wandb-save-dir ${WANDB_PATH}
)


{
    torchrun \
        ${DISTRIBUTED_ARGS[@]} \
        $PRETRAIN_SCRIPT \
        ${DATA_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${PARALLEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${LOGGER_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        --eval-interval 500 \
        --eval-iters 10
} 2>&1 | tee -a "$LOG_FILE"
