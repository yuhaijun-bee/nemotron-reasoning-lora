from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from reasoning_nemotron.paths import RAW_DATA_DIR, SAMPLES_DATA_DIR


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def tag_answer_type(value: object) -> str:
    text = "".join(str(value).strip().split())

    if re.fullmatch(r"[01]+", text) and len(text) >= 8:
        return "binary_number"
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return "number"
    if re.fullmatch(r"[A-Za-z]+", text):
        return "alpha"
    if re.search(r"\d", text):
        return "char_number"
    return "char"


def annotate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["answer_type"] = annotated["answer"].map(tag_answer_type)
    annotated["prompt_len"] = annotated["prompt"].map(len)
    return annotated


def build_stratified_sample(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size >= len(frame):
        return frame.copy()

    _, sampled = train_test_split(
        frame,
        test_size=sample_size,
        stratify=frame["answer_type"],
        random_state=random_state,
    )
    return sampled.sort_values("id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small stratified sample for quick experiments.")
    parser.add_argument("--input", default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--output", default=SAMPLES_DATA_DIR / "train_sample_500.csv")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = annotate_frame(load_data(Path(args.input)))
    sample = build_stratified_sample(frame, args.sample_size, args.random_state)
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(output_path)
    print(sample["answer_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
