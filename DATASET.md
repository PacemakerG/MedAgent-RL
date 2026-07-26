# 数据集说明

## SFT 数据（冷启动训练）

| 文件 | 行数 | 用途 |
|---|---|---|
| `data/MTMedDialog_sft_train.parquet` | 5,516 | SFT 训练 |
| `data/MTMedDialog_sft_val.parquet` | 少量 | SFT 验证 |

**每行结构：**
- `prompt` — 系统指令（医生角色、规则、输出格式）+ 患者自述
- `response` — 医生应有的完整回答（包含 think 推理和 answer）
- `reward_model` — 包含 `ground_truth`（诊断+建议）和 `patient_information`（多轮问诊历史）
- `data_source` — 标识来源

**作用**：让基础模型学会医生问诊的格式、推理方式和多轮对话结构。

## RL 数据（强化学习训练）

| 文件 | 行数 | 用途 |
|---|---|---|
| `data/MTMedDialog_RL.parquet` | 7,068 | GRPO 训练 |

**与 SFT 数据的区别：**
- `prompt` 中嵌入了完整奖励规则（+1 有效提问，-2 重复/格式错误，-5 超轮，+10 诊断正确）
- `reward_model` 多了 `enhanced_description` — 患者完整自述，用于 Patient Agent 模拟回答
- 没有 `response` 字段 — 因为 RL 阶段需要 Doctor 现场生成，没有标准答案

**训练方式**：同一个病例让 Doctor 生成多条问诊轨迹 → 环境计算每条奖励 → GRPO 比较轨迹优劣 → 更新 Doctor 模型参数。

## 测试数据

**文件**：`data/MTMedDialog_test.json`（2,082 条）

**每行结构：**
- `self_report` — 患者自述（简短）
- `enhanced_description` — 患者自述（详细，用于 Patient 模拟）
- `dialogue` — 参考问诊对话（Doctor → Patient 多轮）
- `diagnosis` / `recommendation` — 标准诊断和建议（评价依据）
- `category` — 疾病分类

**作用**：提供病例 + 标准答案，用于评估强化学习后的 Doctor 模型效果。

## 关键对比

| 维度 | SFT 数据 | RL 数据 | 测试数据 |
|---|---|---|---|
| 数据量 | 5,516 | 7,068 | 2,082 |
| 有无标准回答 | ✅ 有（response） | ❌ 无（现场生成） | ❌ 无（用来评） |
| 有无患者自述 | 有（简短） | 有（详细 enhanced_description） | 有 |
| 多久一次 | SFT 冷启动 | 迭代替换 | 固定不变 |
| 多轮对话记录 | 有（patient_information） | 有 | 有（作为参考） |
