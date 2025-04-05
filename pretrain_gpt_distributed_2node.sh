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

export NCCL_BLOCKING_WAIT=1                  # 启用阻塞等待，减少 NCCL 错误传播
export NCCL_ASYNC_ERROR_HANDLING=1           # 开启异步错误处理，提高稳定性
export NCCL_IB_DISABLE=1                     # 禁用 InfiniBand（如果你使用的是 Ethernet）
#export NCCL_DEBUG=INFO                      # 打印 NCCL 调试信息（可根据需要开启）
export NCCL_PORT_RANGE=50000-51000           # 指定 NCCL 可用的端口范围
export NCCL_SOCKET_IFNAME=eno1               # 指定 NCCL 通信使用的网卡名（根据你的机器设置）
export CUDA_DEVICE_MAX_CONNECTIONS=1         # 限制 GPU 的最大连接数，控制通信连接复杂度
#export NCCL_DEBUG_SUBSYS=ALL                # 开启更细粒度的 NCCL 子系统调试（可选）


# 主节点的 IP 地址
MASTER_ADDR=203.91.121.251  # 主节点的 IP 地址
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
CHECKPOINT_PATH=/workspace/model/megatron-models/345m     # 模型保存目录
VOCAB_FILE=/workspace/data/gpt2-vocab.json                # 词表文件
MERGE_FILE=/workspace/data/gpt2-merges.txt                # merge 文件
DATA_PATH=/workspace/data/my-gpt2_text_document           # 训练数据路径
mkdir -p $CHECKPOINT_PATH

# 分布式训练参数
DISTRIBUTED_ARGS="\
    --nproc_per_node $GPUS_PER_NODE \        # 每个节点使用的GPU数
    --nnodes $NNODES \                       # 总节点数
    --node_rank $NODE_RANK \                 # 当前节点编号
    --master_addr $MASTER_ADDR \             # 主节点 IP
    --master_port $MASTER_PORT \             # 主节点端口
"
# GPT训练参数
GPT_ARGS="\
    --num-layers 4 \                         # Transformer 层数
    --hidden-size 256 \                      # 每层隐藏单元数
    --num-attention-heads 8 \                # 注意力头数
    --seq-length 1024 \                      # 输入序列长度
    --max-position-embeddings 1024 \         # 最大位置编码数
    --micro-batch-size 8 \                   # 单 GPU 上的 batch size
    --global-batch-size 64 \                 # 总 batch size（自动拆分）
    --lr 0.00015 \                           # 初始学习率
    --train-iters 300 \                      # 训练迭代次数
    --lr-decay-iters 320000 \                # 学习率衰减步数
    --lr-decay-style cosine \                # 学习率衰减策略
    --min-lr 1.0e-5 \                        # 最小学习率
    --weight-decay 1e-2 \                    # 权重衰减
    --lr-warmup-fraction .01 \               # 学习率预热比例
    --clip-grad 1.0 \                        # 梯度裁剪
    --fp16 \                                 # 使用混合精度
    --tensor-model-parallel-size 2 \         # 张量并行
    --pipeline-model-parallel-size 2 \       # 流水并行
"

# 数据相关参数
DATA_ARGS="\
    --data-path $DATA_PATH \                 # 数据路径
    --vocab-file $VOCAB_FILE \               # 词表文件
    --merge-file $MERGE_FILE \               # merge 文件
    --data-impl mmap \                       # 数据读取方式（内存映射）
    --split 949,50,1 \                       # 训练/验证/测试划分比例
"

# 输出参数
OUTPUT_ARGS="\
    --log-interval 10 \                      # 日志打印间隔
    --save-interval 10000 \                  # 保存模型间隔
    --eval-interval 1000 \                   # 评估间隔
    --eval-iters 10 \                        # 每次评估迭代数
"

# 使用 torchrun 启动分布式训练
torchrun $DISTRIBUTED_ARGS /workspace/megatron-lm-custom/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH
#    --load $CHECKPOINT_PATH
