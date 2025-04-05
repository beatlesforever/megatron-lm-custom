#!/bin/bash

PORT=2345
echo "🔍 检查端口 $PORT 是否被占用..."

PID_LIST=$(lsof -t -i :$PORT)

if [ -n "$PID_LIST" ]; then
    echo "⚠️ 端口 $PORT 已被以下进程占用：$PID_LIST"
    echo "⛔ 尝试自动杀掉这些进程..."

    for pid in $PID_LIST; do
        pname=$(ps -p $pid -o comm=)
        if [[ "$pname" == "torchrun" || "$pname" == "python" ]]; then
            echo "✅ 正在终止进程 $pid ($pname)"
            kill -9 $pid
        else
            echo "⚠️ 进程 $pid ($pname) 非 torchrun/python，不自动结束，请手动处理！"
        fi
    done
else
    echo "✅ 端口 $PORT 未被占用。"
fi

sleep 1

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_PORT_RANGE=50000-51000
export NCCL_SOCKET_IFNAME=eno1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 主节点的 IP 地址
MASTER_ADDR=203.91.121.252  # 主节点的 IP 地址
MASTER_PORT=2345

# 设置每台机器上的 GPU 数量
GPUS_PER_NODE=2

# 当前机器的节点排名（第二节点）
NODE_RANK=1

# 参与训练的总节点数
NNODES=2

# 计算总的 world size
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# 模型保存和加载路径
CHECKPOINT_PATH=/workspace/model/megatron-models/345m
VOCAB_FILE=/workspace/data/gpt2-vocab.json
MERGE_FILE=/workspace/data/gpt2-merges.txt
DATA_PATH=/workspace/data/my-gpt2_text_document

mkdir -p $CHECKPOINT_PATH

# 分布式训练参数
DISTRIBUTED_ARGS="\
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# GPT训练参数
GPT_ARGS="\
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 300 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
"

# 数据相关参数
DATA_ARGS="\
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
"

# 输出参数
OUTPUT_ARGS="\
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

# 使用 torchrun 启动分布式训练
torchrun $DISTRIBUTED_ARGS /workspace/megatron-lm-custom/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
