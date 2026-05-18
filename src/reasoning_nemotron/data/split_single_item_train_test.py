from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reasoning_nemotron.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


TRAIN_STATUS = "derived_base"


def build_splits(refined_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_ids = set(refined_df.loc[refined_df["final_rewrite_status"] == TRAIN_STATUS, "id"])

    refined_train = refined_df[refined_df["id"].isin(train_ids)].copy()
    refined_test = refined_df[~refined_df["id"].isin(train_ids)].copy()
    raw_train = raw_df[raw_df["id"].isin(train_ids)].copy()
    raw_test = raw_df[~raw_df["id"].isin(train_ids)].copy()

    refined_train["split"] = "train"
    refined_test["split"] = "test"
    raw_train["split"] = "train"
    raw_test["split"] = "test"

    return {
        "refined_train": refined_train,
        "refined_test": refined_test,
        "raw_train": raw_train,
        "raw_test": raw_test,
    }


def summary_payload(refined_train: pd.DataFrame, refined_test: pd.DataFrame, total_rows: int) -> dict[str, object]:
    train_count = len(refined_train)
    test_count = len(refined_test)
    return {
        "rule": f"{TRAIN_STATUS} => train, others => test",
        "total_rows": total_rows,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_ratio": train_count / total_rows if total_rows else 0.0,
        "test_ratio": test_count / total_rows if total_rows else 0.0,
        "train_by_question_type": refined_train["question_type"].value_counts().sort_index().to_dict(),
        "test_by_question_type": refined_test["question_type"].value_counts().sort_index().to_dict(),
        "test_by_final_status": refined_test["final_rewrite_status"].value_counts().sort_index().to_dict(),
        "test_by_ref_category": refined_test["ref_category"].value_counts().sort_index().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the 9500-row dataset into single-item strict train and test sets.")
    parser.add_argument(
        "--raw-input",
        default=RAW_DATA_DIR / "train.csv",
    )
    parser.add_argument(
        "--refined-input",
        default=PROCESSED_DATA_DIR / "train_reasoning_refined.csv",
    )
    parser.add_argument(
        "--raw-train-output",
        default=PROCESSED_DATA_DIR / "train_single_item_strict_train_raw.csv",
    )
    parser.add_argument(
        "--raw-test-output",
        default=PROCESSED_DATA_DIR / "train_single_item_strict_test_raw.csv",
    )
    parser.add_argument(
        "--refined-train-output",
        default=PROCESSED_DATA_DIR / "train_single_item_strict_train_refined.csv",
    )
    parser.add_argument(
        "--refined-test-output",
        default=PROCESSED_DATA_DIR / "train_single_item_strict_test_refined.csv",
    )
    parser.add_argument(
        "--summary-output",
        default=PROCESSED_DATA_DIR / "train_single_item_strict_split_summary.json",
    )
    args = parser.parse_args()

    raw_df = pd.read_csv(Path(args.raw_input), dtype=str).fillna("")
    refined_df = pd.read_csv(Path(args.refined_input), dtype=str).fillna("")

    if len(raw_df) != len(refined_df):
        raise ValueError(f"Row count mismatch: raw={len(raw_df)} refined={len(refined_df)}")

    split_frames = build_splits(refined_df, raw_df)

    split_frames["raw_train"].to_csv(Path(args.raw_train_output), index=False, encoding="utf-8-sig")
    split_frames["raw_test"].to_csv(Path(args.raw_test_output), index=False, encoding="utf-8-sig")
    split_frames["refined_train"].to_csv(Path(args.refined_train_output), index=False, encoding="utf-8-sig")
    split_frames["refined_test"].to_csv(Path(args.refined_test_output), index=False, encoding="utf-8-sig")

    summary = summary_payload(
        split_frames["refined_train"],
        split_frames["refined_test"],
        len(refined_df),
    )
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(Path(args.raw_train_output))
    print(Path(args.raw_test_output))
    print(Path(args.refined_train_output))
    print(Path(args.refined_test_output))
    print(Path(args.summary_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
