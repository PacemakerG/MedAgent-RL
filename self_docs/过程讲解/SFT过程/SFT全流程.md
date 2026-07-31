> torchrun 启动多进程 → 加载 Qwen → PEFT 注入 LoRA → FSDP 分片 → DistributedSampler 分数据 → Qwen 做 next-token prediction → loss_mask 屏蔽问题 → 反传并只更新 LoRA → 验证、保存 adapter。

## 0. SFT 启动入口

医学 SFT 从这里启动：

[finetune_lora_med.sh (line 13)](/Users/elon2ge/workspace/MedAgent-RL/sft/finetune_lora_med.sh:13)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m ragen.trainer.fsdp_sft_trainer
```

关键配置在：

[finetune_lora_med.sh (line 15)](/Users/elon2ge/workspace/MedAgent-RL/sft/finetune_lora_med.sh:15)

```bash
data.train_files=data/MTMedDialog_sft_train.parquet
data.val_files=data/MTMedDialog_sft_val.parquet
model.partial_pretrain=Qwen2.5-7B-Instruct
model.lora_rank=64
model.lora_alpha=32
model.target_modules=all-linear
```

完整调用链：

```text
finetune_lora_med.sh
  → main()
  → FSDPSFTTrainer.__init__()
  → _build_dataloader()
  → _build_model_optimizer()
  → fit()
  → training_step()
  → _compute_loss()
```

## 1. 加载 Qwen 模型

[fsdp_sft_trainer.py (line 177)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:177)

```python
self.model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    torch_dtype=torch.float32,
    attn_implementation='flash_attention_2',
)
```

这里的 `local_model_path` 来自：

```bash
model.partial_pretrain=Qwen2.5-7B-Instruct
```

也就是：

```text
Qwen2.5-7B-Instruct
        ↓
AutoModelForCausalLM.from_pretrained()
```

## 2. 给 Qwen 加入 LoRA

[fsdp_sft_trainer.py (line 183)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:183)

```python
if self.config.model.get('lora_rank', 0) > 0:
    lora_config = {
        'task_type': TaskType.CAUSAL_LM,
        'r': self.config.model.lora_rank,
        'lora_alpha': self.config.model.lora_alpha,
        'target_modules': convert_to_regular_types(
            self.config.model.target_modules
        ),
        'bias': "none"
    }
    self.model = get_peft_model(
        self.model,
        LoraConfig(**lora_config)
    )
```

医学脚本配置为：

```text
rank = 64
alpha = 32
target_modules = all-linear
```

`get_peft_model()` 会冻结 Qwen 原始参数，将 LoRA 参数设为可训练参数。因此后面虽然优化器接收整个模型，真正产生梯度并被更新的是 LoRA 参数。

## 3. 用 FSDP 将模型切到多张 GPU

首先初始化多 GPU 进程和 device mesh：

[fsdp_sft_trainer.py (line 430)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:430)

```python
local_rank, rank, world_size = initialize_global_process_group()

device_mesh = init_device_mesh(
    device_type='cuda',
    mesh_shape=(world_size,),
    mesh_dim_names=('dp',)
)
```

然后使用 FSDP 包装模型：

[fsdp_sft_trainer.py (line 215)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:215)

```python
self.fsdp_model = FSDP(
    module=self.model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision,
    device_mesh=self.device_mesh,
    device_id=torch.cuda.current_device(),
)
```

核心是：

```python
sharding_strategy=ShardingStrategy.FULL_SHARD
```

它负责分片：

- 模型参数
- 梯度
- 优化器状态

## 4. 每张 GPU 获取自己的训练样本

[fsdp_sft_trainer.py (line 130)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:130)

```python
rank = self.device_mesh.get_rank()
world_size = self.device_mesh.size()

self.train_sampler = DistributedSampler(
    self.train_dataset,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    drop_last=True
)
```

DataLoader 使用这个 sampler：

[fsdp_sft_trainer.py (line 138)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:138)

```python
self.train_dataloader = DataLoader(
    dataset=self.train_dataset,
    sampler=self.train_sampler,
    batch_size=config.data.train_batch_size,
)
```

准确地说，这里是：

```text
每个进程都加载完整 Dataset 到 CPU 内存
        ↓
DistributedSampler 给每个 rank 分配不同的数据索引
        ↓
每张 GPU 实际训练不同样本
```

完整 parquet 的加载代码在：

[sft_dataset.py (line 92)](/Users/elon2ge/workspace/MedAgent-RL/ragen/utils/dataset/sft_dataset.py:92)

## 5. 将问题和回答拼成训练序列

[sft_dataset.py (line 135)](/Users/elon2ge/workspace/MedAgent-RL/ragen/utils/dataset/sft_dataset.py:135)

```python
prompt_chat_str, response_chat_str = apply_chat_template(
    tokenizer,
    prompt,
    response,
    with_thinking=self.with_thinking,
    add_generation_prompt=True,
    tokenize=False
)

response_chat_str += tokenizer.eos_token
```

分别 tokenize 后拼接：

[sft_dataset.py (line 138)](/Users/elon2ge/workspace/MedAgent-RL/ragen/utils/dataset/sft_dataset.py:138)

```python
prompt_ids = tokenizer(prompt_chat_str, ...)
response_ids = tokenizer(response_chat_str, ...)

input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
```

最终输入类似：

```text
[用户问题 token] [assistant 起始标记] [回答 token] [EOS]
```

## 6. 模型预测下一个 token

模型前向传播：

[fsdp_sft_trainer.py (line 253)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:253)

```python
output = self.fsdp_model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    position_ids=batch['position_ids'],
    use_cache=False
)

logits = output.logits
```

通过错位实现 next-token prediction：

[fsdp_sft_trainer.py (line 250)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:250)

```python
labels = batch['input_ids'][:, 1:]
shift_logits = logits[..., :-1, :]
```

对应关系：

```text
位置 t 的 logits  →  预测位置 t+1 的 token
```

这是 teacher forcing：训练时完整正确答案已经在 `input_ids` 中。

## 7. 只对回答部分计算 loss

首先在 Dataset 中生成 `loss_mask`：

[sft_dataset.py (line 177)](/Users/elon2ge/workspace/MedAgent-RL/ragen/utils/dataset/sft_dataset.py:177)

```python
loss_mask = attention_mask.clone()

if prompt_length > 1:
    # 屏蔽 prompt
    loss_mask[:prompt_length - 1] = 0

# 屏蔽回答最后一个无法继续预测的位置
loss_mask[prompt_length + response_length - 1] = 0
```

效果：

```text
问题部分：0 0 0 0
回答部分：1 1 1 1
Padding： 0 0 0
```

再将逐 token loss 乘上 mask：

[fsdp_sft_trainer.py (line 264)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:264)

```python
loss_fct = nn.CrossEntropyLoss(reduction='none')
loss = loss_fct(shift_logits, shift_labels)
loss = loss * loss_mask
loss = torch.sum(loss) / torch.sum(loss_mask)
```

因此问题 token 和 padding token 的 loss 都变成了 0。

## 8. 反向传播

[fsdp_sft_trainer.py (line 292)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:292)

```python
micro_batches = batch.split(self.config.data.micro_batch_size)

for micro_batch in micro_batches:
    loss = self._compute_loss(micro_batch) / n_micro_batches
    loss.backward()
```

这里还做了梯度累积：一个大 batch 被拆成多个 micro batch，每个 micro batch 都调用一次 `backward()`。

随后裁剪梯度：

```python
self.fsdp_model.clip_grad_norm_(
    max_norm=self.config.optim.clip_grad
)
```

## 9. 更新 LoRA 参数

优化器创建：

[fsdp_sft_trainer.py (line 228)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:228)

```python
self.optimizer = optim.AdamW(
    self.fsdp_model.parameters(),
    lr=self.config.optim.lr,
    betas=self.config.optim.betas,
    weight_decay=self.config.optim.weight_decay
)
```

实际更新：

[fsdp_sft_trainer.py (line 304)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:304)

```python
self.optimizer.step()
self.lr_scheduler.step()
```

因为 PEFT 已经冻结 Qwen 原始参数，所以效果是：

- Qwen 原始参数：不更新
- LoRA A/B 矩阵：更新

## 10. 验证模型

验证时关闭梯度：

[fsdp_sft_trainer.py (line 319)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:319)

```python
self.fsdp_model.eval()

with torch.no_grad():
    loss = self._compute_loss(batch)
    torch.distributed.all_reduce(
        loss,
        op=torch.distributed.ReduceOp.AVG
    )
```

每个 epoch 结束后的完整验证循环：

[fsdp_sft_trainer.py (line 406)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:406)

## 11. 保存 LoRA checkpoint

[fsdp_sft_trainer.py (line 326)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:326)

```python
with FSDP.state_dict_type(
    self.fsdp_model,
    StateDictType.FULL_STATE_DICT,
    cfg
):
    state_dict = self.fsdp_model.state_dict()

if rank == 0:
    self.model.save_pretrained(path, state_dict=state_dict)
    self.tokenizer.save_pretrained(path)
```

保存目录类似：

```text
$save_path/global_step_100/
$save_path/global_step_200/
```

训练主循环中“验证 → 保存”对应：

[fsdp_sft_trainer.py (line 379)](/Users/elon2ge/workspace/MedAgent-RL/ragen/trainer/fsdp_sft_trainer.py:379)

```python
for epoch in range(total_epochs):
    for data in train_dataloader:
        metric = self.training_step(data)

    # validation
    ...

    # save checkpoint
    self.save_checkpoint(step=global_step)
```

## 最终代码版流程图

```text
torchrun 启动 N 个进程
    ↓
AutoModelForCausalLM.from_pretrained() 加载 Qwen
    ↓
get_peft_model() 注入 LoRA、冻结原参数
    ↓
FSDP(... FULL_SHARD) 分片参数/梯度/优化器状态
    ↓
DistributedSampler(rank=rank) 给每卡分配样本
    ↓
prompt_ids + response_ids 拼成 input_ids
    ↓
logits[:, :-1] 预测 input_ids[:, 1:]
    ↓
CrossEntropyLoss(reduction="none")
    ↓
loss *= loss_mask，只保留回答 token
    ↓
loss.backward()
    ↓
optimizer.step()，实际更新 LoRA
    ↓
no_grad() 验证
    ↓
save_pretrained() 保存 LoRA checkpoint
```
