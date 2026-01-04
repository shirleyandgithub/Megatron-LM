#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=1
PP=2
GPUS_PER_NODE=2

DATA_PATH=/root/data/my-gpt2_text_document
TOKENIZER_PATH=/root/data/gpt2_local

# Global Batch Size = 1，意为强制气泡最大化
torchrun --nproc_per_node $GPUS_PER_NODE \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers 12 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 100 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --use-cpu-initialization \
    --transformer-impl local \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --eval-iters 0
