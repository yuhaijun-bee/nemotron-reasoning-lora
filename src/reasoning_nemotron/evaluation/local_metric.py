from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def extract_boxed_answer(text: object) -> str:
    raw = str(text).strip()
    matches = re.findall(r"\\boxed\{([^}]*)\}", raw)
    return matches[-1].strip() if matches else raw


def normalize_text(text: object) -> str:
    return str(text).strip()


def verify(stored_answer: object, predicted: object) -> bool:
    gold = extract_boxed_answer(normalize_text(stored_answer))
    pred = extract_boxed_answer(normalize_text(predicted))

    if re.fullmatch(r"[01]+", gold):
        return pred.lower() == gold.lower()

    try:
        return math.isclose(float(gold), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except ValueError:
        return pred.lower() == gold.lower()


def score_list(answers: list[str], predictions: list[str]) -> float:
    if len(answers) != len(predictions):
        raise ValueError("The number of answers and predictions must match.")
    if not answers:
        return 0.0
    return sum(verify(a, p) for a, p in zip(answers, predictions)) / len(answers)


def score_predictions(solution_df: pd.DataFrame, prediction_df: pd.DataFrame, row_id_col: str) -> float:
    merged = solution_df[[row_id_col, "answer"]].merge(
        prediction_df[[row_id_col, "prediction"]],
        on=row_id_col,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping ids were found between labels and predictions.")
    return sum(verify(a, p) for a, p in zip(merged["answer"], merged["prediction"])) / len(merged)


def compare_predictions(solution_df: pd.DataFrame, prediction_df: pd.DataFrame, row_id_col: str) -> pd.DataFrame:
    merged = solution_df[[row_id_col, "answer"]].merge(
        prediction_df[[row_id_col, "prediction"]],
        on=row_id_col,
        how="inner",
    ).copy()
    merged["is_correct"] = merged.apply(
        lambda row: verify(row["answer"], row["prediction"]),
        axis=1,
    )
    return merged[[row_id_col, "answer", "prediction", "is_correct"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a prediction CSV against the gold answers.")
    parser.add_argument("--solution", required=True, help="CSV with id and answer columns.")
    parser.add_argument("--prediction", required=True, help="CSV with id and prediction columns.")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--output", default=None, help="Optional CSV path for row-level comparison.")
    args = parser.parse_args()

    solution_df = pd.read_csv(Path(args.solution))
    prediction_df = pd.read_csv(Path(args.prediction))

    accuracy = score_predictions(solution_df, prediction_df, args.id_col)
    print(f"accuracy={accuracy:.4f}")

    if args.output:
        comparison = compare_predictions(solution_df, prediction_df, args.id_col)
        comparison.to_csv(Path(args.output), index=False, encoding="utf-8-sig")
        print(Path(args.output))


if __name__ == "__main__":
    main()
