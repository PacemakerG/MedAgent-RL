"""Extract readable samples from the MedAgent-RL Parquet datasets."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


DATASETS = {
    "rl": "MTMedDialog_RL.parquet",
    "sft_train": "MTMedDialog_sft_train.parquet",
    "sft_val": "MTMedDialog_sft_val.parquet",
}


def read_first_records(parquet_path: Path, limit: int) -> tuple[int, list[dict]]:
    parquet_file = parquet.ParquetFile(parquet_path)
    records: list[dict] = []

    for batch in parquet_file.iter_batches(batch_size=limit):
        records.extend(batch.to_pylist())
        if len(records) >= limit:
            break

    return parquet_file.metadata.num_rows, records[:limit]


def export_dataset(
    parquet_path: Path,
    output_dir: Path,
    limit: int,
) -> Path:
    _, records = read_first_records(parquet_path, limit)
    output_path = output_dir / f"{parquet_path.stem}_first_{limit}.json"

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Extract the first records from RL, SFT train, and SFT val."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to extract from each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "extracted_samples",
        help="Directory for the readable JSON files.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than zero")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, filename in DATASETS.items():
        parquet_path = repository_root / "data" / filename
        output_path = export_dataset(
            parquet_path=parquet_path,
            output_dir=output_dir,
            limit=args.limit,
        )
        print(f"{dataset_name}: {output_path}")


if __name__ == "__main__":
    main()
