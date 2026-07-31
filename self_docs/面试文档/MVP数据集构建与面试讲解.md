# 医疗对话数据集构建 MVP

## 目标

用最小规模走通 DoctorAgent-RL 数据构建主线：

```text
三种原始格式
→ 字段映射
→ 文本清洗与角色统一
→ 多轮质量过滤
→ 固定随机抽样
→ 病例级 SFT / RL / holdout 划分
→ SFT turn 数据与 RL case 数据
→ LLM 增强任务
```

这是一套透明、可复现的教学 MVP，不声称复刻论文未公开的完整预处理代码。

## 运行方式

在仓库根目录执行：

```bash
python3 self_scripts/build_mvp_dataset.py
```

默认参数：

- 每个来源抽样 50 个病例；
- 随机种子为 42；
- 每个来源按病例划分 20% SFT、60% RL、20% holdout；
- MedDG 暂时使用 `dev.txt`，因为本地缺少 `train.txt`。

## 三种原始格式

| 来源 | 原始格式 | 病例组织方式 | 标签特点 |
|---|---|---|---|
| IMCS21 | JSON | 顶层字典中的每个值是一个病例 | 有诊断、报告和建议，属于直接病例级标签 |
| CHIP-MDCFNPC | JSONL | 每行一个对话 | 主要是 clinical finding NER，疾病实体只能作为弱标签 |
| MedDG | 自定义 TXT | `dialogN` 开始一个病例，后续每行是一个 utterance JSON | 每轮带症状、药物、检查、属性、疾病实体 |

## 统一病例结构

```json
{
  "case_id": "imcs21:train-001",
  "source": "IMCS21",
  "source_split": "train",
  "language": "zh",
  "self_report": "患者初始自述",
  "dialogue": [
    {
      "turn_id": 0,
      "role": "patient",
      "text": "对话文本",
      "entities": []
    }
  ],
  "ground_truth": {
    "diagnosis": "诊断或弱标签候选",
    "recommendation": "建议或 null",
    "label_quality": "gold | weak | missing"
  },
  "quality": {
    "doctor_turn_count": 4,
    "patient_turn_count": 4,
    "sft_ready": true,
    "rl_ready": true
  },
  "mvp_split": "sft | rl | holdout"
}
```

## 清洗和质量门槛

1. 统一空白符，去掉空 utterance。
2. 将不同来源中的角色统一为 `doctor`、`patient`、`unknown`。
3. 合并连续的同角色 utterance。
4. 使用内容哈希去除完全重复病例。
5. 只保留至少 3 个 Doctor turn 且至少 3 个 Patient turn 的病例。
6. SFT-ready：存在患者自述，并且至少有 Doctor 和 Patient 内容。
7. RL-ready：在上述条件之外，还必须有可靠的病例级诊断和建议。

第 7 条非常重要。CHIP 和 MedDG 虽然能抽出疾病实体，但它们不是与 IMCS21 等价的病例级诊断/建议金标准，不能直接当作强化学习奖励。

## 本次实际结果

| 产物 | 数量 | 含义 |
|---|---:|---|
| 统一病例 | 150 | 三个来源各 50 |
| SFT 病例 | 30 | 三个来源各 10 |
| SFT turn 样本 | 334 | 每个 Doctor turn 形成一个监督样本 |
| RL 病例种子 | 90 | 三个来源各 30 |
| RL-ready 病例 | 30 | 当前只有 IMCS21 的 30 条具备直接诊断和建议 |
| holdout 病例 | 30 | 三个来源各 10 |
| Cold-start 输入病例 | 30 | 每病例一次请求，覆盖 334 个 Doctor turn |

## 输出文件

```text
data/processed_samples/
├── mvp_unified_cases.jsonl
├── mvp_sft_seed_turns.jsonl
├── mvp_rl_seed_cases.jsonl
├── mvp_holdout_cases.jsonl
├── mvp_cold_start_input.json
└── mvp_manifest.json
```

### SFT seed

每个原始 Doctor utterance 都变成一个监督样本：

```text
系统指令 + 患者自述 + 当前历史 → 原始医生下一步回复
```

当前 `thinking_status=not_generated`。`mvp_cold_start_input.json` 将同一病例
的 Doctor turn 合并为一次请求；模型只返回 `turn_id + thinking`，原始答案由
本地脚本保真拼装。

### Cold-start thinking

在 `self_scripts/generate_mvp_cold_start_sft.py` 顶部填写：

```python
API_KEY = "..."
BASE_URL = "..."
MODEL_NAME = "..."
```

安装 OpenAI 兼容客户端后运行：

```bash
python3 -m pip install openai
python3 self_scripts/generate_mvp_cold_start_sft.py
```

脚本默认使用 4 个 worker。全部病例成功后输出
`mvp_cold_start_sft.json`，每条数据只保留 SFT 训练需要的两个字段：

```text
prompt + response
```

`response` 使用 `<think>...</think><answer>...</answer>` 格式。三套中文原始
数据中的 Doctor utterance 会原样放入 `<answer>`；脚本不会强行改写为
`Question:`，避免把同时包含建议的回答误标为问题。

### RL seed

每个病例是一个环境种子：

```text
Doctor 初始 prompt
+ ground truth
+ 患者自述
+ 参考对话
```

当前 `enhanced_description_status=not_generated`。下一阶段才使用 Qwen2.5-7B-Instruct 生成 Patient Agent 的隐藏完整病例画像。

## 面试时可以这样讲

> 我先分析三套原始数据的粒度和标注差异。IMCS21 是病例级 JSON，有诊断和建议；CHIP 是每行一个对话的 JSONL，重点是临床实体；MedDG 是自定义文本分块，每轮带实体标注。我没有直接拼接，而是先定义统一病例模型，统一角色、对话和实体字段，再进行多轮过滤、内容去重和固定种子分层抽样。划分在病例级完成，因此 SFT、RL、holdout 不会共享同一病例。SFT 数据按 Doctor turn 展开，而 RL 保持 case-level，用作在线交互环境。最后设置 label-quality gate：只有具备可靠诊断和建议的病例才能直接进入 RL，弱标签数据需要先经过模型抽取和人工复核。

## 与论文完整流程的差距

- 尚未把中文内容翻译成英文；
- 尚未调用 DeepSeek-V3 生成 Doctor thinking；
- 尚未调用 Qwen2.5-7B-Instruct 生成隐藏患者画像；
- MedDG 使用的是 dev split，不是缺失的 train split；
- 作者对三源字段的完整映射、噪声识别 prompt 和人工复核规范没有公开。

SFT thinking 阶段的确定性输入已经写入 `mvp_cold_start_input.json`。RL 标签复核、
患者画像生成和 train/test 划分属于后续独立步骤。
