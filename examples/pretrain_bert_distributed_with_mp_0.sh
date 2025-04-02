#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CHECKPOINT_PATH=<Specify path>
# VOCAB_FILE=<Specify path to file>/bert-vocab.txt
# DATA_PATH=<Specify path and file prefix>_text_sentence
CHECKPOINT_PATH=/workspace/model/bert/345m-1f1b-4
VOCAB_FILE=/workspace/data/bert/bert-vocab.txt
DATA_PATH=/workspace/data/bert/corpus_indexed_text_sentence
MODEL_PATH=/workspace/model/bert/output
REPORT_PATH=/workspace/model/bert/report

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --pipeline-1f1b-enable 0 \
    --num-layers 48 \
    --hidden-size 2560 \
    --num-attention-heads 32 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --lr 0.0001 \
    --train-iters 1000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
    --tokenizer-type BertWordPieceCase
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

TIMING_ARGS="
    --timing-log-level 2
    --timing-log-option all
"

WRITER_ARGS="
    --tensorboard-dir $REPORT_PATH
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $TIMING_ARGS \
    $OUTPUT_ARGS \
    $WRITER_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
