# MedAgent-RL：一周复现与 Mind Agent 融合计划

## 仓库组织

保持两个仓库独立，不互相复制整仓库：

```text
workspace/
├── MedAgent-RL/          # SFT、GRPO、训练与评测
└── HardWare-Medicial/    # Mind Agent、RAG、ECG、前后端应用
```

原因：

- 保留 `MedAgent-RL` 的 fork 历史，方便同步上游。
- 训练环境与 FastAPI/LangGraph 应用环境分离。
- 最终只传递模型权重、推理接口和评测结果，不传递整个训练仓库。

## 一周任务

### Day 1：环境与基线

- 执行 `scripts/setup_ragen.sh`。
- 下载基础模型、SFT 模型、RL 模型和数据集。
- 跑通官方推理与评测脚本。
- 保存 Base、SFT、RL 三组基线结果。

### Day 2：代码链路

看清以下调用链：

```text
Doctor Agent
→ Patient Agent
→ Medical Consultation Environment
→ Reward / Evaluator
→ GRPO rollout 与参数更新
```

重点确认：状态格式、动作格式、结束条件、动态轮数和奖励组成。

### Day 3：SFT 复现

- 检查 `MTMedDialog_sft_train.parquet`。
- 跑通 LoRA SFT。
- 保存 checkpoint、loss 和推理样例。
- 对比 Base 与 SFT 的问诊效果。

### Day 4：GRPO 复现

- 检查 `MTMedDialog_RL.parquet`。
- 跑通 Doctor、Patient、Evaluator 的完整 rollout。
- 确认 reward、advantage 和模型参数真实更新。
- 保存 RL checkpoint 和训练日志。

### Day 5：评测与复现结果

对比：

- Base Model
- SFT Model
- SFT + GRPO Model

记录：

- 诊断准确率
- 平均问诊轮数
- 关键症状覆盖率
- 无效追问比例
- 对话完成率

### Day 6：接入 Mind Agent

在 `HardWare-Medicial` 中新增：

```text
rl_doctor_policy
├── 输入：对话历史、已知症状、RAG/ECG 工具结果
└── 输出：继续追问、调用 RAG、调用 ECG、结束并诊断
```

接入方式：

1. `MedAgent-RL` 提供本地模型推理服务或模型加载接口。
2. `HardWare-Medicial` 通过统一 `DoctorPolicyClient` 调用。
3. 保留原有安全分诊，高风险情况直接绕过 RL Agent。
4. RAG 和 ECG 继续作为 Mind Agent 工具，不搬进训练仓库。

核心流程：

```text
患者输入
→ 安全分诊
→ RL Doctor Policy
→ 追问 / RAG / ECG / 最终诊断
→ 更新问诊状态
→ 下一轮
```

### Day 7：端到端交付

- 跑通 Mind Agent 多轮问诊 Demo。
- 完成 Base、SFT、RL 三组对比。
- 补充一键启动脚本。
- 整理架构图、训练日志、评测表格和 README。

## 两个仓库的边界

### MedAgent-RL

负责：

- 数据处理
- SFT
- GRPO
- Patient Agent
- Reward / Evaluator
- 模型评测
- 模型权重导出

### HardWare-Medicial

负责：

- LangGraph 工作流
- 用户会话与状态管理
- 医疗安全分诊
- 科室 RAG
- ECG 工具
- RL Doctor Policy 推理调用
- 前端与 API

## 完成标准

- 官方推理和评测能够运行。
- 完成一次 LoRA SFT。
- 完成一次真实 GRPO 参数更新。
- 有 Base、SFT、RL 三组结果。
- Mind Agent 能自主追问、调用工具并结束问诊。
- 两个仓库可以独立安装和运行。