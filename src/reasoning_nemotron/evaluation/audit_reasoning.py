from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_nemotron.data.annotate_reasoning import roman_with_steps
from reasoning_nemotron.paths import PROCESSED_DATA_DIR


JUMP_PHRASES = {
    "Text encryption conversion": [
        "labeled resolution is",
        "labeled plaintext phrase",
    ],
    "Binary operation conversion": [
        "labeled output is",
    ],
    "Character equation conversion": [
        "Applying the labeled rule",
    ],
    "Numerical equations conversion": [
        "Applying the labeled rule",
    ],
    "Digital character operation": [
        "Applying the labeled rule",
    ],
}


def audit_roman(row: pd.Series) -> list[str]:
    issues: list[str] = []
    prompt = row["prompt"]
    answer = row["answer"]
    reasoning = row["reasoning"]
    target = int(re.search(r"Now, write the number (\d+)", prompt).group(1))
    computed, _ = roman_with_steps(target)
    if computed != answer:
        issues.append(f"computed_roman_mismatch:{computed}")
    if f"gives {computed}" not in reasoning and f"gives {answer}" not in reasoning:
        issues.append("missing_explicit_computed_result")
    return issues


def audit_gravity(row: pd.Series) -> list[str]:
    issues: list[str] = []
    prompt = row["prompt"]
    answer = float(row["answer"])
    pairs = [
        (float(t), float(d))
        for t, d in re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    ]
    target_t = float(
        re.search(r"Now, determine the falling distance for t = ([\d.]+)s", prompt).group(1)
    )
    g_values = [2 * d / (t * t) for t, d in pairs]
    avg_g = sum(g_values) / len(g_values)
    pred = 0.5 * avg_g * target_t * target_t
    if abs(pred - answer) > 0.06:
        issues.append(f"answer_numeric_mismatch:{pred:.4f}")
    return issues


def audit_unit(row: pd.Series) -> list[str]:
    issues: list[str] = []
    prompt = row["prompt"]
    answer = float(row["answer"])
    pairs = [(float(x), float(y)) for x, y in re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)]
    target_x = float(
        re.search(r"convert the following measurement: ([\d.]+) m", prompt).group(1)
    )
    xs = np.array([x for x, _ in pairs], dtype=float)
    ys = np.array([y for _, y in pairs], dtype=float)
    design = np.vstack([xs, np.ones(len(xs))]).T
    slope, intercept = np.linalg.lstsq(design, ys, rcond=None)[0]
    pred = slope * target_x + intercept
    if abs(pred - answer) > 0.06:
        issues.append(f"answer_numeric_mismatch:{pred:.4f}")
    return issues


def audit_text(row: pd.Series) -> list[str]:
    issues: list[str] = []
    reasoning = row["reasoning"]
    for phrase in JUMP_PHRASES["Text encryption conversion"]:
        if phrase in reasoning:
            issues.append(f"jump_step_via_label:{phrase}")
    return issues


def audit_binary(row: pd.Series) -> list[str]:
    issues: list[str] = []
    reasoning = row["reasoning"]
    for phrase in JUMP_PHRASES["Binary operation conversion"]:
        if phrase in reasoning:
            issues.append(f"jump_step_via_label:{phrase}")
    return issues


def audit_equation(row: pd.Series) -> list[str]:
    issues: list[str] = []
    question_type = row["question_type"]
    reasoning = row["reasoning"]
    for phrase in JUMP_PHRASES[question_type]:
        if phrase in reasoning:
            issues.append(f"jump_step_via_label:{phrase}")
    return issues


def audit_row(row: pd.Series) -> list[str]:
    issues: list[str] = []

    if not row["reasoning"].endswith(f"answer : {row['answer']}"):
        issues.append("answer_suffix_mismatch")

    qtype = row["question_type"]
    if qtype == "Digital counting conversion":
        issues.extend(audit_roman(row))
    elif qtype == "Gravity conversion":
        issues.extend(audit_gravity(row))
    elif qtype == "Unit conversion":
        issues.extend(audit_unit(row))
    elif qtype == "Text encryption conversion":
        issues.extend(audit_text(row))
    elif qtype == "Binary operation conversion":
        issues.extend(audit_binary(row))
    elif qtype in {
        "Character equation conversion",
        "Numerical equations conversion",
        "Digital character operation",
    }:
        issues.extend(audit_equation(row))
    else:
        issues.append("unknown_question_type")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the reasoning annotations row by row.")
    parser.add_argument(
        "--input",
        default=PROCESSED_DATA_DIR / "train_reasoning_strict_labeled.csv",
    )
    parser.add_argument(
        "--output",
        default=PROCESSED_DATA_DIR / "train_reasoning_audit.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input), dtype=str).fillna("")
    issue_lists = [audit_row(row) for _, row in df.iterrows()]

    audit = df[["id", "question_type"]].copy()
    audit["has_issue"] = [bool(items) for items in issue_lists]
    audit["issue_count"] = [len(items) for items in issue_lists]
    audit["issues"] = [" | ".join(items) for items in issue_lists]
    audit.to_csv(Path(args.output), index=False, encoding="utf-8-sig")

    print(Path(args.output))
    print(audit["has_issue"].value_counts().to_string())
    print(audit[audit["has_issue"]]["question_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
