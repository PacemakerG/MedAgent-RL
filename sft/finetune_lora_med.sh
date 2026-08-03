#!/usr/bin/env bash

set -euo pipefail
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 <num_gpus> <save_path> [hydra overrides...]"
    exit 1
fi

# Reduce allocator fragmentation for long-context training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$1
save_path=$2

mkdir -p "$save_path"

shift 2
# 已在单节点 8 张 24 GB NVIDIA RTX 4090 D 上完成实测。
# micro_batch_size=8 归一化后为每张 GPU 处理 1 个样本，实测显存约为 12-14 GB/卡；
# 可尝试 micro_batch_size=16（每卡 2 个样本）提高吞吐，但必须先做一步 OOM 测试。
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=data/MTMedDialog_sft_train.parquet \
    data.val_files=data/MTMedDialog_sft_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    optim.lr=1e-4 \
    data.train_batch_size=128 \
    data.micro_batch_size=8 \
    +data.with_thinking=False \
    model.partial_pretrain=Qwen2.5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=med_dialogue-sft-thinking-lora-Qwen2.5-7B-Instruct \
    trainer.project_name=Medical-Dialogue \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null $@ \
    trainer.validate_before_training=True \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    2>&1 | tee "$save_path/train.log"

echo "SFT finished. Select a checkpoint under $save_path/global_step_* before merging LoRA weights."
