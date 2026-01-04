#!/bin/bash

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 并行策略
TP=2
PP=1
GPUS_PER_NODE=2

# 数据路径 (不需要加.bin或.idx后缀，Megatron会自动找)
DATA_PATH=/root/data/my-gpt2_text_document

# 指向造好的本地Tokenizer目录
TOKENIZER_PATH=/root/data/gpt2_local

DATA_PATH=/root/data/my-gpt2_text_document
TOKENIZER_PATH=/root/data/gpt2_local

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
    --global-batch-size 8 \
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
