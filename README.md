# DoctorAgent-RL：多智能体协同强化学习临床对话系统

[论文 arXiv](https://arxiv.org/pdf/2505.19630) | [HuggingFace 模型](https://huggingface.co/collections/Jarvis1111/doctoragent-rl-684ffbcade52305ba0e3e97f)

## 一句话

用多智能体 + GRPO 强化学习训练医生大模型，让它学会在多次问诊中主动收集信息、给出诊断和建议。

## 核心架构

三个角色协作：

1. **Doctor Agent** — 问诊的医生，负责提问和最终诊断。**这是被训练的模型。**
2. **Patient Agent** — 模拟患者，根据给定病例回答医生问题，不透露诊断。
3. **咨询评价器（Consultation Evaluator）** — 给每次问诊打分：有效提问加分，重复/无效/超轮扣分，诊断正确大加分。

训练流程：Doctor 和 Patient 多轮对话 → 评价器算奖励 → GRPO 算法比较同病例不同问法 → 更新 Doctor 策略。

## 训练流程

### 1. 数据
- `data/MTMedDialog_sft_train.parquet` — SFT 冷启动数据（5,516 条）
- `data/MTMedDialog_RL.parquet` — RL 训练数据（7,068 条，含患者详细自述）
- `data/MTMedDialog_test.json` — 测试集（2,082 条）

### 2. 环境搭建
```bash
git clone https://github.com/JarvisUSTC/DoctorAgent-RL.git
cd DoctorAgent-RL
bash scripts/setup_ragen.sh
```

### 3. 下载模型
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（基座模型）
- [DoctorAgent-RL-SFT-1k-Thinking](https://huggingface.co/Jarvis1111/DoctorAgent-RL-SFT-1k-Thinking)（作者的 SFT 模型）
- [DoctorAgent-RL](https://huggingface.co/Jarvis1111/DoctorAgent-RL)（作者的 RL 训练后模型）

### 4. 训练命令
```bash
# SFT 冷启动
bash sft/finetune_lora_med.sh

# 完整 RL（含 SFT 冷启动 + 动态轮数 + 奖励模型）
bash scripts_exp/doctor-agent-rl-rm-dynamic.sh

# 变体
bash scripts_exp/doctor-agent-rl-dynamic.sh           # 不含奖励模型
bash scripts_exp/doctor-agent-rl-rm.sh                # 固定轮数
bash scripts_exp/doctor-agent-rl-dynamic-wo-sft.sh    # 不经过 SFT
```

### 5. 评测
```bash
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh ${MODEL_PATH}
```

## 引用
```bibtex
@article{feng2025doctoragent,
  title={DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue},
  author={Feng, Yichun and Wang, Jiawei and Zhou, Lu and Li, Yixue},
  journal={arXiv preprint arXiv:2505.19630},
  year={2025}
}
```
