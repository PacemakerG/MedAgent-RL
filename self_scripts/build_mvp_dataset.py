#!/usr/bin/env python3
"""Build a deterministic, interview-friendly MTMedDialog MVP dataset.

This script does not claim to reproduce the authors' unpublished preprocessing.
It demonstrates the inspectable part of the pipeline:

raw source formats -> normalized cases -> quality gates -> case-level splits
-> SFT turn seeds / RL case seeds -> case-level cold-start input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


PIPELINE_VERSION = "1.0"
SOURCE_ORDER = ("IMCS21", "CHIP-MDCFNPC", "MedDG")

DOCTOR_SYSTEM_PROMPT = (
    "You are a doctor conducting a multi-turn consultation. Ask one focused "
    "question at a time. When enough evidence is available, provide a diagnosis "
    "and recommendation."
)


def clean_text(value: object) -> str:
    """Normalize whitespace while preserving human-readable content."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_role(value: object) -> str:
    role = clean_text(value).lower()
    if "医生" in role or "doctor" in role:
        return "doctor"
    if "患者" in role or "病人" in role or "patient" in role:
        return "patient"
    return "unknown"


def unique_strings(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def merge_adjacent_turns(turns: list[dict]) -> list[dict]:
    """Merge consecutive utterances from the same speaker."""
    merged: list[dict] = []
    for turn in turns:
        text = clean_text(turn.get("text"))
        if not text:
            continue
        role = normalize_role(turn.get("role"))
        entities = turn.get("entities", [])
        if merged and merged[-1]["role"] == role:
            merged[-1]["text"] = f"{merged[-1]['text']}\n{text}"
            merged[-1]["entities"].extend(entities)
        else:
            merged.append(
                {
                    "turn_id": len(merged),
                    "role": role,
                    "text": text,
                    "entities": list(entities),
                }
            )
    return merged


def case_fingerprint(case: dict) -> str:
    payload = {
        "self_report": clean_text(case.get("self_report")),
        "dialogue": [
            {"role": turn["role"], "text": clean_text(turn["text"])}
            for turn in case["dialogue"]
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_case(
    *,
    source: str,
    source_split: str,
    source_record_id: object,
    self_report: object,
    raw_turns: list[dict],
    diagnosis: object = None,
    recommendation: object = None,
    label_quality: str = "missing",
) -> dict:
    dialogue = merge_adjacent_turns(raw_turns)
    initial_report = clean_text(self_report)
    if not initial_report:
        initial_report = next(
            (turn["text"] for turn in dialogue if turn["role"] == "patient"),
            "",
        )

    diagnosis_text = clean_text(diagnosis)
    recommendation_text = clean_text(recommendation)
    doctor_turns = sum(turn["role"] == "doctor" for turn in dialogue)
    patient_turns = sum(turn["role"] == "patient" for turn in dialogue)
    has_three_exchange_turns = doctor_turns >= 3 and patient_turns >= 3

    case = {
        "case_id": f"{source.lower().replace('-', '_')}:{source_record_id}",
        "source": source,
        "source_split": source_split,
        "source_record_id": str(source_record_id),
        "language": "zh",
        "self_report": initial_report,
        "dialogue": dialogue,
        "ground_truth": {
            "diagnosis": diagnosis_text or None,
            "recommendation": recommendation_text or None,
            "label_quality": label_quality,
        },
        "quality": {
            "raw_utterance_count": len(raw_turns),
            "merged_turn_count": len(dialogue),
            "doctor_turn_count": doctor_turns,
            "patient_turn_count": patient_turns,
            "has_self_report": bool(initial_report),
            "has_three_exchange_turns": has_three_exchange_turns,
            "sft_ready": bool(initial_report and doctor_turns and patient_turns),
            "rl_ready": bool(
                initial_report
                and has_three_exchange_turns
                and diagnosis_text
                and recommendation_text
                and label_quality == "gold"
            ),
        },
    }
    case["content_fingerprint"] = case_fingerprint(case)
    return case


def load_imcs21(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for record_id, record in records.items():
        raw_turns: list[dict] = []
        for utterance in record.get("dialogue", []):
            symptom_names = utterance.get("symptom_norm") or []
            symptom_types = utterance.get("symptom_type") or []
            entities = [
                {
                    "type": "Symptom",
                    "name": clean_text(name),
                    "attribute": clean_text(symptom_types[index])
                    if index < len(symptom_types)
                    else None,
                }
                for index, name in enumerate(symptom_names)
                if clean_text(name)
            ]
            raw_turns.append(
                {
                    "role": utterance.get("speaker"),
                    "text": utterance.get("sentence"),
                    "entities": entities,
                }
            )

        reports = [item for item in record.get("report", []) if isinstance(item, dict)]
        selected_report = reports[-1] if reports else {}
        cases.append(
            finalize_case(
                source="IMCS21",
                source_split="train",
                source_record_id=record_id,
                self_report=record.get("self_report"),
                raw_turns=raw_turns,
                diagnosis=record.get("diagnosis") or selected_report.get("诊断"),
                recommendation=selected_report.get("建议"),
                label_quality="gold",
            )
        )
    return cases


def load_chip_mdcfnpc(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            raw_turns: list[dict] = []
            disease_candidates: list[str] = []
            for utterance in record.get("dialog_info", []):
                entities = []
                for entity in utterance.get("ner", []):
                    entity_type = clean_text(entity.get("type"))
                    entity_name = clean_text(entity.get("name"))
                    if entity_type.lower() == "disease" and entity_name != "undefined":
                        disease_candidates.append(entity_name)
                    entities.append(
                        {
                            "type": entity_type or None,
                            "name": entity_name or None,
                            "mention": clean_text(entity.get("mention")) or None,
                            "attribute": clean_text(entity.get("attr")) or None,
                        }
                    )
                raw_turns.append(
                    {
                        "role": utterance.get("sender"),
                        "text": utterance.get("text"),
                        "entities": entities,
                    }
                )
            diseases = unique_strings(disease_candidates)
            cases.append(
                finalize_case(
                    source="CHIP-MDCFNPC",
                    source_split="train",
                    source_record_id=record.get("dialog_id"),
                    self_report=next(
                        (
                            turn.get("text")
                            for turn in raw_turns
                            if normalize_role(turn.get("role")) == "patient"
                        ),
                        "",
                    ),
                    raw_turns=raw_turns,
                    diagnosis="；".join(diseases),
                    recommendation=None,
                    label_quality="weak" if diseases else "missing",
                )
            )
    return cases


def iter_meddg_dialogues(path: Path) -> Iterator[tuple[str, list[dict]]]:
    current_id: str | None = None
    utterances: list[dict] = []
    with path.open(encoding="utf-8-sig") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("dialog"):
                if current_id is not None:
                    yield current_id, utterances
                current_id = line
                utterances = []
            else:
                utterances.append(json.loads(line))
    if current_id is not None:
        yield current_id, utterances


def load_meddg(path: Path, source_split: str) -> list[dict]:
    cases: list[dict] = []
    entity_fields = ("Symptom", "Medicine", "Examination", "Attribute", "Disease")
    for dialogue_id, utterances in iter_meddg_dialogues(path):
        raw_turns: list[dict] = []
        disease_candidates: list[str] = []
        for utterance in utterances:
            entities = []
            for field in entity_fields:
                for value in utterance.get(field, []):
                    name = clean_text(value)
                    if not name:
                        continue
                    entities.append(
                        {"type": field, "name": name, "attribute": None}
                    )
                    if field == "Disease":
                        disease_candidates.append(name)
            raw_turns.append(
                {
                    "role": utterance.get("id"),
                    "text": utterance.get("Sentence"),
                    "entities": entities,
                }
            )
        diseases = unique_strings(disease_candidates)
        cases.append(
            finalize_case(
                source="MedDG",
                source_split=source_split,
                source_record_id=dialogue_id,
                self_report=next(
                    (
                        turn.get("text")
                        for turn in raw_turns
                        if normalize_role(turn.get("role")) == "patient"
                    ),
                    "",
                ),
                raw_turns=raw_turns,
                diagnosis="；".join(diseases),
                recommendation=None,
                label_quality="weak" if diseases else "missing",
            )
        )
    return cases


def stable_random(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def select_cases(
    source_cases: dict[str, list[dict]],
    sample_per_source: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    selected: list[dict] = []
    used_fingerprints: set[str] = set()
    source_stats: dict[str, dict[str, int]] = {}

    for source in SOURCE_ORDER:
        raw_cases = source_cases[source]
        eligible = [
            case
            for case in raw_cases
            if case["quality"]["has_self_report"]
            and case["quality"]["has_three_exchange_turns"]
        ]
        unique_candidates: list[dict] = []
        local_fingerprints: set[str] = set()
        for case in eligible:
            fingerprint = case["content_fingerprint"]
            if (
                fingerprint not in local_fingerprints
                and fingerprint not in used_fingerprints
            ):
                local_fingerprints.add(fingerprint)
                unique_candidates.append(case)
        if len(unique_candidates) < sample_per_source:
            raise ValueError(
                f"{source} only has {len(unique_candidates)} eligible unique cases; "
                f"requested {sample_per_source}"
            )

        sample = stable_random(seed, source).sample(
            unique_candidates, sample_per_source
        )
        sample.sort(key=lambda item: item["case_id"])
        selected.extend(sample)
        used_fingerprints.update(case["content_fingerprint"] for case in sample)
        source_stats[source] = {
            "available": len(raw_cases),
            "eligible_after_cleaning": len(eligible),
            "eligible_unique": len(unique_candidates),
            "sampled": len(sample),
        }

    return selected, source_stats


def assign_case_splits(cases: list[dict], seed: int) -> list[dict]:
    """Assign 20% SFT, 60% RL, and 20% holdout within each source."""
    output: list[dict] = []
    for source in SOURCE_ORDER:
        source_cases = [case for case in cases if case["source"] == source]
        stable_random(seed, f"split:{source}").shuffle(source_cases)
        sft_size = round(len(source_cases) * 0.2)
        holdout_size = round(len(source_cases) * 0.2)
        rl_boundary = len(source_cases) - holdout_size
        for index, case in enumerate(source_cases):
            case_copy = dict(case)
            if index < sft_size:
                case_copy["mvp_split"] = "sft"
            elif index < rl_boundary:
                case_copy["mvp_split"] = "rl"
            else:
                case_copy["mvp_split"] = "holdout"
            output.append(case_copy)
    return sorted(output, key=lambda item: item["case_id"])


def build_sft_turns(cases: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for case in cases:
        if case["mvp_split"] != "sft" or not case["quality"]["sft_ready"]:
            continue
        history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]
        doctor_index = 0
        for turn in case["dialogue"]:
            if turn["role"] == "doctor":
                if len(history) == 1:
                    history.append(
                        {
                            "role": "user",
                            "content": f"Patient self-report: {case['self_report']}",
                        }
                    )
                examples.append(
                    {
                        "example_id": f"{case['case_id']}:doctor:{doctor_index}",
                        "case_id": case["case_id"],
                        "source": case["source"],
                        "prompt": list(history),
                        "response": turn["text"],
                        "response_origin": "original_doctor_utterance",
                        "thinking_status": "not_generated",
                    }
                )
                history.append({"role": "assistant", "content": turn["text"]})
                doctor_index += 1
            elif turn["role"] == "patient":
                history.append({"role": "user", "content": turn["text"]})
    return examples


def build_rl_seeds(cases: list[dict]) -> list[dict]:
    seeds: list[dict] = []
    for case in cases:
        if case["mvp_split"] != "rl":
            continue
        seeds.append(
            {
                "case_id": case["case_id"],
                "source": case["source"],
                "prompt": [
                    {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Patient self-report: {case['self_report']}",
                    },
                ],
                "ground_truth": case["ground_truth"],
                "patient_profile_seed": {
                    "self_report": case["self_report"],
                    "reference_dialogue": case["dialogue"],
                },
                "enhanced_description": None,
                "enhanced_description_status": "not_generated",
                "rl_ready": case["quality"]["rl_ready"],
            }
        )
    return seeds


def build_cold_start_input(cases: list[dict]) -> list[dict]:
    """Keep only the case-level fields required to generate Doctor thinking."""
    return [
        {
            "case_id": case["case_id"],
            "self_report": case["self_report"],
            "dialogue": [
                {
                    "turn_id": turn["turn_id"],
                    "role": turn["role"],
                    "text": turn["text"],
                }
                for turn in case["dialogue"]
            ],
        }
        for case in cases
        if case["mvp_split"] == "sft" and case["quality"]["sft_ready"]
    ]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, rows: Iterable[dict]) -> int:
    materialized = list(rows)
    path.write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    repository_root: Path,
    output_dir: Path,
    sample_per_source: int = 50,
    seed: int = 42,
    meddg_split: str = "dev",
) -> dict:
    if sample_per_source < 5:
        raise ValueError("sample_per_source must be at least 5")
    if meddg_split not in {"train", "dev", "test"}:
        raise ValueError("meddg_split must be train, dev, or test")

    inputs = {
        "IMCS21": repository_root
        / "data/raw_sources/IMCS21/dataset/train.json",
        "CHIP-MDCFNPC": repository_root
        / "data/raw_sources/CHIP-MDCFNPC/CHIP-MDCFNPC/"
        "CHIP-MDCFNPC_train.jsonl",
        "MedDG": repository_root
        / f"data/raw_sources/MedDG/MedDG/data/read/{meddg_split}.txt",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))

    source_cases = {
        "IMCS21": load_imcs21(inputs["IMCS21"]),
        "CHIP-MDCFNPC": load_chip_mdcfnpc(inputs["CHIP-MDCFNPC"]),
        "MedDG": load_meddg(inputs["MedDG"], meddg_split),
    }
    selected, source_stats = select_cases(source_cases, sample_per_source, seed)
    cases = assign_case_splits(selected, seed)
    sft_turns = build_sft_turns(cases)
    rl_seeds = build_rl_seeds(cases)
    holdout_cases = [case for case in cases if case["mvp_split"] == "holdout"]
    cold_start_cases = build_cold_start_input(cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "unified_cases": output_dir / "mvp_unified_cases.jsonl",
        "sft_turns": output_dir / "mvp_sft_seed_turns.jsonl",
        "rl_seeds": output_dir / "mvp_rl_seed_cases.jsonl",
        "holdout_cases": output_dir / "mvp_holdout_cases.jsonl",
        "cold_start_input": output_dir / "mvp_cold_start_input.json",
    }
    output_counts = {
        "unified_cases": write_jsonl(outputs["unified_cases"], cases),
        "sft_turns": write_jsonl(outputs["sft_turns"], sft_turns),
        "rl_seeds": write_jsonl(outputs["rl_seeds"], rl_seeds),
        "holdout_cases": write_jsonl(outputs["holdout_cases"], holdout_cases),
        "cold_start_input": write_json(
            outputs["cold_start_input"], cold_start_cases
        ),
    }

    split_counts = Counter(case["mvp_split"] for case in cases)
    source_split_counts = Counter(
        (case["source"], case["mvp_split"]) for case in cases
    )
    case_ids = [case["case_id"] for case in cases]
    fingerprints = [case["content_fingerprint"] for case in cases]
    rl_ready_counts = Counter(
        (seed_row["source"], str(seed_row["rl_ready"])) for seed_row in rl_seeds
    )
    checks = {
        "expected_unified_cases": sample_per_source * len(SOURCE_ORDER),
        "actual_unified_cases": len(cases),
        "unique_case_ids": len(set(case_ids)),
        "unique_content_fingerprints": len(set(fingerprints)),
        "case_level_split_overlap": False,
        "all_outputs_nonempty": all(value > 0 for value in output_counts.values()),
    }
    if checks["actual_unified_cases"] != checks["expected_unified_cases"]:
        raise AssertionError("Unexpected unified case count")
    if checks["unique_case_ids"] != len(cases):
        raise AssertionError("Duplicate case IDs detected")
    if checks["unique_content_fingerprints"] != len(cases):
        raise AssertionError("Duplicate normalized cases detected")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "purpose": "transparent MVP; not an exact reproduction of unpublished preprocessing",
        "parameters": {
            "sample_per_source": sample_per_source,
            "seed": seed,
            "meddg_split": meddg_split,
            "split_policy": "20% SFT / 60% RL / 20% holdout per source",
            "minimum_dialogue_rule": "at least 3 merged doctor turns and 3 merged patient turns",
        },
        "inputs": {
            source: str(path.relative_to(repository_root))
            for source, path in inputs.items()
        },
        "source_stats": source_stats,
        "output_counts": output_counts,
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": {
            f"{source}:{split}": count
            for (source, split), count in sorted(source_split_counts.items())
        },
        "rl_readiness": {
            f"{source}:{ready}": count
            for (source, ready), count in sorted(rl_ready_counts.items())
        },
        "checks": checks,
        "known_limitations": [
            "MedDG dev is used because train.txt is not present locally.",
            "Only IMCS21 provides direct case-level diagnosis and recommendation labels.",
            "CHIP-MDCFNPC and MedDG disease entities are weak label candidates, not gold rewards.",
            "Translation and DeepSeek thinking generation are not executed by this builder.",
        ],
        "output_sha256": {
            name: sha256_file(path) for name, path in outputs.items()
        },
    }
    manifest_path = output_dir / "mvp_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build a deterministic 3-source medical-dialogue MVP dataset."
    )
    parser.add_argument("--sample-per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--meddg-split",
        choices=("train", "dev", "test"),
        default="dev",
        help="Use dev until the missing MedDG train.txt is downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "data/processed_samples",
    )
    args = parser.parse_args()

    manifest = build_dataset(
        repository_root=repository_root,
        output_dir=args.output_dir.resolve(),
        sample_per_source=args.sample_per_source,
        seed=args.seed,
        meddg_split=args.meddg_split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
