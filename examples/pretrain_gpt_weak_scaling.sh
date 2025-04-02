#!/bin/bash

# Weak scaling
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8

VOCAB_FILE=/workspace/model/gpt2-vocab/gpt2-vocab.json
MERGE_FILE=/workspace/model/gpt2-vocab/gpt2-merges.txt
DATA_PATH=/workspace/data/bert/corpus_indexed_text_sentence

hidden_sizes=(1536 1920 2304 3072)
attention_heads=(16 20 24 32)
num_layers=(40 54 64 72)
num_params=(1.2 2.5 4.2 8.3)
model_parallel_gpus=(1 2 4 8)
data_parallel_gpus=(64 128 256 512)

for i in ${!hidden_sizes[@]}; do
    HIDDEN_SIZE=${hidden_sizes[$i]}
    ATTENTION_HEADS=${attention_heads[$i]}
    NUM_LAYERS=${num_layers[$i]}
    NUM_PARAMS=${num_params[$i]}
    MODEL_PARALLEL=${model_parallel_gpus[$i]}

    echo "Running configuration:"
    echo "Number of Parameters: $NUM_PARAMS"
    echo "Model Parallel GPUs: $MODEL_PARALLEL"
    
    GPUS_PER_NODE=${MODEL_PARALLEL}
    # Change for multinode config
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))    

    CHECKPOINT_PATH=/workspace/model/gpt2/${NUM_PARAMS}B-MP${MODEL_PARALLEL}
    MODEL_PATH=/workspace/model/gpt2/output
    REPORT_PATH=/workspace/model/gpt2/report

    DISTRIBUTED_ARGS="
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    GPT_ARGS="
        --tensor-model-parallel-size $MODEL_PARALLEL \
        --pipeline-model-parallel-size 1 \
        --sequence-parallel \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $ATTENTION_HEADS \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 4 \
        --global-batch-size 16 \
        --lr 0.00015 \
        --train-iters 100 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --lr-warmup-fraction .01 \
        --clip-grad 1.0 \
        --fp16
    "

    DATA_ARGS="
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
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

    # nsys profile -o $REPORT_PATH --trace=cuda,nvtx,osrt,cudnn,cublas \
    torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $TIMING_ARGS \
        $OUTPUT_ARGS \
        $WRITER_ARGS \
        --distributed-backend nccl \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH

    echo "Completed training for the current configuration."
done

