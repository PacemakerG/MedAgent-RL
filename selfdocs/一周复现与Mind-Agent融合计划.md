# MedAgent-RL：一周复现与 Mind Agent 融合计划

## 一、仓库与机器分工

保持两个仓库独立，并在 Mac 和集群上都平级放置：

```text
workspace/
├── MedAgent-RL/          # 数据、SFT、GRPO、Patient Agent、奖励与评测
└── HardWare-Medicial/    # Mind Agent、LangGraph、RAG、ECG、前后端
```

不要把一个完整仓库复制进另一个仓库。

### Mac 本地负责

- 拉取两个仓库和 `verl` 子模块。
- 阅读数据、训练脚本和核心源码。
- 修改配置、代码、文档并提交 Git。
- 查看小规模数据样例和训练日志。
- 不下载 7B 模型，不安装 CUDA 训练环境，不保存大型 checkpoint。

```bash
cd ~/workspace
git clone --recurse-submodules git@github.com:PacemakerG/MedAgent-RL.git
git clone git@github.com:PacemakerG/HardWare-Medicial.git
```

如果 `MedAgent-RL` 已经 clone：

```bash
cd ~/workspace/MedAgent-RL
git submodule update --init --recursive
```

### GPU 集群负责

- 拉取 `MedAgent-RL` 和 `verl` 子模块。
- 创建 Conda、CUDA、PyTorch、vLLM、Ray、FSDP 环境。
- 下载 Qwen 基础模型和可选的作者 checkpoint。
- 运行 SFT、GRPO、评测和模型合并。
- 保存模型权重、训练日志、WandB 记录和评测结果。
- 第六天接入时，再拉取 `HardWare-Medicial`。

```bash
git clone --recurse-submodules git@github.com:PacemakerG/MedAgent-RL.git
cd MedAgent-RL
bash scripts/setup_ragen.sh
```

### 模型下载原则

从头复现时，集群首先只需要下载：

```text
Qwen/Qwen2.5-7B-Instruct
```

下面两个作者模型只用于验证和对比，不是必须的训练起点：

```text
Jarvis1111/DoctorAgent-RL-SFT-1k-Thinking
Jarvis1111/DoctorAgent-RL
```

目标训练路径：

```text
Qwen2.5-7B-Instruct
→ 自己运行 LoRA SFT
→ 自己的 SFT checkpoint
→ GRPO
→ 自己的 MedAgent-RL 模型
```

## 二、需要阅读的仓库

### 1. PacemakerG/MedAgent-RL

主仓库。重点理解项目如何把医疗问诊做成 SFT 和强化学习任务。

### 2. volcengine/verl

位于 `MedAgent-RL/verl/`，是 Git 子模块。重点理解：

- Actor、Reference Policy、Critic 的组织方式。
- FSDP Worker 和分布式资源分配。
- rollout、log probability、advantage 和 policy loss。
- GRPO 如何比较同一问题的多条轨迹。

不需要通读整个仓库。使用下面的命令定位核心实现：

```bash
rg -n "grpo|advantage|actor_loss|policy_loss|log_prob" verl ragen
```

### 3. RAGEN 相关代码

当前已经包含在 `MedAgent-RL/ragen/` 中，不需要再单独 clone。重点看 Agent 环境如何接入 `verl` 训练循环。

### 4. PacemakerG/HardWare-Medicial

第六天再重点阅读，用于把训练后的 Doctor Policy 接入 Mind Agent。

## 三、一周任务与源码阅读顺序

### Day 1：数据、目录和环境

#### Mac 阅读

- `README.md`：项目目标与官方运行方式。
- `DATASET.md`：SFT、RL、测试集的用途。
- `data/MTMedDialog_sft_train.parquet`：确认 `prompt`、`response` 字段。
- `data/MTMedDialog_RL.parquet`：确认病例、患者描述、真实诊断和建议。
- `.gitmodules`：确认 `verl` 子模块。
- `scripts/setup_ragen.sh`：确认环境安装过程。
- `requirements.txt`：确认主要依赖。

#### 集群执行

- clone 仓库和子模块。
- 创建训练环境。
- 下载 `Qwen2.5-7B-Instruct`。
- 检查 GPU、CUDA、磁盘和模型加载。

### Day 2：看懂并跑通 SFT

#### Mac 阅读

按顺序看：

1. `sft/finetune_lora_med.sh`
   - 数据路径。
   - 基础模型路径。
   - batch、长度、学习率、LoRA 参数。
2. `ragen/trainer/fsdp_sft_trainer.py`
   - 数据加载。
   - tokenizer 和 prompt/response 拼接。
   - loss 计算。
   - FSDP、梯度累积和 checkpoint 保存。
3. `sft/utils/merge_lora.py`
   - LoRA adapter 如何合并回基础模型。

需要讲清楚：

```text
一条 SFT 数据如何变成 token
→ 哪些 token 参与 loss
→ LoRA 参数如何更新
→ checkpoint 如何保存和合并
```

#### 集群执行

- 先用少量数据跑通 LoRA SFT。
- 确认 loss 下降和 checkpoint 生成。
- 再运行正式可承受配置。

### Day 3：看懂 Doctor、Patient 和奖励环境

#### Mac 阅读

按顺序看：

1. `ragen/env/medical_consultation/env.py`
   - 基础问诊状态、动作和终止条件。
2. `ragen/env/medical_consultation/env_patient_llm.py`
   - Patient Agent 如何读取隐藏病例。
   - Doctor 的问题如何交给 Patient Agent。
   - 重复问题、无效问题和基础奖励。
3. `ragen/env/medical_consultation/env_patient_llm_rm.py`
   - LLM 如何给诊断和建议打分。
   - Reward Model 版本与规则版本的区别。
4. `ragen/workers/env_llm_worker.py`
   - Patient Agent 和评价模型如何批量推理。

需要讲清楚：

```text
Doctor 输出动作
→ 环境解析动作
→ Patient 生成回复
→ 环境更新对话状态
→ 计算奖励
→ 判断继续还是结束
```

#### 集群执行

- 单独运行一批 rollout。
- 保存完整问诊轨迹。
- 人工核对每一步 observation、action、reward 和 done。

### Day 4：看懂并跑通 GRPO

#### Mac 阅读

按顺序看：

1. `scripts_exp/doctor-agent-rl-rm-dynamic.sh`
   - 8 卡、batch、rollout 数量、动态轮数和模型配置。
2. `ragen/trainer/main_ppo.py`
   - 环境注册、RewardManager、Ray Worker 创建和训练入口。
3. `ragen/trainer/ppo/ray_trainer.py`
   - rollout、奖励汇总、advantage 和参数更新主循环。
4. `ragen/workers/fsdp_workers.py`
   - Actor、Reference Policy 和 Critic 的前向、反向与更新。
5. `verl/` 中通过搜索定位：
   - GRPO advantage。
   - log probability。
   - PPO/GRPO policy loss。
   - KL 与 reference policy。

需要讲清楚：

```text
同一病例生成多条问诊轨迹
→ 每条轨迹得到总奖励
→ 组内标准化得到 advantage
→ 计算新旧策略概率比
→ 计算 GRPO loss
→ 只更新 Doctor Agent
```

#### 集群执行

- 把官方 8 卡配置改成当前集群可用配置。
- 先用小数据、小 batch、少 rollout 跑通一次真实参数更新。
- 确认 reward、advantage、loss 和模型参数都发生变化。

### Day 5：评测和结果对比

#### Mac 阅读

- `ragen/env/medical_consultation/evaluation/`
- 评测脚本和 prompt 模板。
- 模型加载、患者模拟和指标统计方式。

#### 集群执行

对比：

- Base：Qwen2.5-7B-Instruct。
- SFT：自己训练的 SFT 模型。
- RL：自己完成 GRPO 的模型。
- 作者 RL：可选参考上限。

记录：

- 诊断得分。
- 建议得分。
- 平均问诊轮数。
- 重复或无效问题比例。
- 格式错误率。
- 对话完成率。

### Day 6：接入 Mind Agent

#### Mac 阅读 HardWare-Medicial

重点目录：

- `backend/app/core/`：状态和 LangGraph 工作流。
- `backend/app/agents/`：现有医疗 Agent 节点。
- `backend/app/services/`：聊天、会话、ECG 等服务。
- `backend/app/tools/`：RAG、搜索和模型调用工具。
- `backend/app/api/v1/endpoints/`：聊天与流式接口。

新增统一调用层：

```text
DoctorPolicyClient
├── 输入：对话历史、患者回复、剩余轮数、工具结果
└── 输出：继续追问或最终诊断
```

边界：

- `MedAgent-RL` 负责训练和模型推理服务。
- `HardWare-Medicial` 负责会话、安全分诊、RAG、ECG 和前端。
- 不把 RAGEN、verl 和训练依赖复制进应用仓库。

#### 集群执行

- 启动训练后模型的推理服务。
- 让 `HardWare-Medicial` 通过 API 调用 Doctor Policy。
- 保留原有安全分诊，高风险情况绕过 RL Agent。

### Day 7：端到端交付

- 跑通多轮问诊 Demo。
- 完成 Base、SFT、RL 对比表。
- 保存训练曲线、典型轨迹和失败案例。
- 补充运行脚本、架构图和 README。
- 整理可以写入简历的技术链路和指标。

## 四、完成标准

- Mac 上能完整定位并解释 SFT、环境、奖励和 GRPO 主调用链。
- 集群上完成一次 LoRA SFT，并生成自己的 checkpoint。
- 集群上完成一次真实 GRPO 参数更新。
- 有 Base、SFT、RL 三组评测结果。
- 能解释为什么 SFT 是冷启动，以及 RL 在 SFT 基础上学到了什么。
- Mind Agent 能调用训练后的 Doctor Policy 完成多轮追问和最终诊断。
- 两个仓库保持独立，可以分别安装、训练和运行。