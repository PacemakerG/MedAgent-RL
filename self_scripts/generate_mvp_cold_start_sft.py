#!/usr/bin/env python3
"""按病例生成 Doctor 的思考过程，并组装最终的 SFT JSON。

模型接收一个完整病例，仅返回：

    {"thinkings": [{"turn_id": 1, "thinking": "..."}]}

原始 prompt 和 Doctor 回复由本地程序组装，最终每条数据只包含
prompt 和 response。
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# 运行前填写以下三个值。不要提交真实的 API 密钥。
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

MAX_WORKERS = 4
TEMPERATURE = 0.2
MAX_TOKENS = 4096


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    REPOSITORY_ROOT / "data/processed_samples/mvp_cold_start_input.json"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT / "data/processed_samples/mvp_cold_start_sft.json"
)

SFT_SYSTEM_PROMPT = """You are an experienced doctor who needs to provide professional diagnosis and advice to patients through consultation. Please listen carefully to the patient's description, ask targeted questions, and collect sufficient information before giving a diagnosis and treatment recommendation.

Quick Guide
Objectives:
1. Obtain key information through effective questioning, each round of questions should be modified based on the previous round's content, meaning you shouldn't ask similar questions.
2. Comprehensively analyze the patient's condition to provide an accurate diagnosis and appropriate treatment recommendations.

Rules:
1. You can only choose one of the options to respond, you cannot both answer questions and provide a diagnosis simultaneously.
2. Absolutely do not repeat or ask questions similar or identical to those previously asked.

Response:
<think> [your thinking] </think>
<answer>If you believe there is insufficient information, please only ask one question, in this format:
Question: (your question).
</answer> | <answer>If you believe you have obtained enough information, please only provide diagnosis and recommendations, in this format:
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or advice)
</answer>
"""

THINKING_SYSTEM_PROMPT = """You generate concise medical reasoning for Doctor turns in a multi-turn consultation.

Requirements:
1. Return JSON only. Do not use Markdown fences.
2. Return exactly one thinking item for every Doctor turn_id.
3. Do not rewrite, quote, or return the Doctor answer.
4. For each Doctor turn, reason only from information available before that turn. Never use later dialogue turns.
5. The thinking should explain why that question, diagnosis, or recommendation is the appropriate next action.
6. Use the same language as the corresponding Doctor message.
7. Do not include <think>, <answer>, or any other XML tags.

Output schema:
{"thinkings":[{"turn_id":1,"thinking":"..."}]}
"""

def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_api_messages(case: dict) -> list[dict]:
    model_input = {
        "case_id": case["case_id"],
        "self_report": case["self_report"],
        "dialogue": case["dialogue"],
    }
    return [
        {"role": "system", "content": THINKING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(model_input, ensure_ascii=False),
        },
    ]


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_case_thinkings(case: dict) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=make_api_messages(case),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    payload = json.loads(response.choices[0].message.content)
    return {
        "case_id": case["case_id"],
        "thinkings": payload["thinkings"],
    }


def build_sft_records(cases: list[dict], results: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    for case in cases:
        thinking_by_turn = {
            item["turn_id"]: item["thinking"]
            for item in results[case["case_id"]]["thinkings"]
        }
        history = [
            {"content": SFT_SYSTEM_PROMPT, "role": "system"},
            {
                "content": (
                    f"Patient's description: {case['self_report'].strip()}\n"
                    "Decide next action:\n"
                    "Always output: <think> [your thinking] </think> "
                    "<answer> [your response] </answer> No additional text. "
                    "Strictly follow this format.\n"
                ),
                "role": "user",
            },
        ]
        doctor_seen = False
        for turn in case["dialogue"]:
            if turn["role"] == "patient":
                text = turn["text"].strip()
                if (
                    not doctor_seen
                    and text == case["self_report"].strip()
                ):
                    continue
                history.append({"content": text, "role": "user"})
                continue
            if turn["role"] != "doctor":
                continue

            doctor_seen = True
            response = (
                f"<think>{thinking_by_turn[turn['turn_id']]}</think>\n"
                f"<answer>{turn['text'].strip()}</answer>"
            )
            records.append(
                {
                    "prompt": [dict(message) for message in history],
                    "response": response,
                }
            )
            history.append({"content": response, "role": "assistant"})
    return records


def process_cases(cases: list[dict]) -> dict[str, dict]:
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(generate_case_thinkings, case)
            for case in cases
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["case_id"]] = result
            print(f"[{index}/{len(cases)}] {result['case_id']}")
    return results


def main() -> None:
    cases = load_cases(INPUT_PATH)
    results = process_cases(cases)
    records = build_sft_records(cases, results)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
