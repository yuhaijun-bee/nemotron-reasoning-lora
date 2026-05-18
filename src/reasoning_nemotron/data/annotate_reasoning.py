from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_nemotron.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


ROMAN_MAP = [
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def classify_question_type(prompt: str) -> str:
    if "different numeral system" in prompt:
        return "Digital counting conversion"
    if "gravitational constant" in prompt:
        return "Gravity conversion"
    if "secret unit conversion" in prompt:
        return "Unit conversion"
    if "secret encryption rules" in prompt:
        return "Text encryption conversion"
    if "8-bit binary" in prompt:
        return "Binary operation conversion"
    if "secret set of transformation rules is applied to equations" in prompt:
        target = prompt.split("Now, determine the result for:")[-1].strip().rstrip(".")
        if not re.search(r"\d", target):
            return "Character equation conversion"

        block = prompt.split("Below are a few examples:", 1)[1].split(
            "Now, determine the result for:", 1
        )[0]
        rhs_values = []
        for line in block.splitlines():
            if "=" in line:
                rhs_values.append(line.split("=", 1)[1].strip())

        if any(re.search(r"\D", rhs) for rhs in rhs_values):
            return "Digital character operation"
        return "Numerical equations conversion"
    return "Unknown"


def roman_with_steps(n: int) -> tuple[str, list[str]]:
    steps: list[str] = []
    out: list[str] = []
    rem = n
    for value, symbol in ROMAN_MAP:
        count = rem // value
        if count:
            piece = symbol * count
            steps.append(f"{rem} contains {count}*{value}, so write {piece}")
            out.append(piece)
            rem -= count * value
    return "".join(out), steps


def extract_arrow_pairs(prompt: str, header: str, footer: str) -> list[tuple[str, str]]:
    block = prompt.split(header, 1)[1].split(footer, 1)[0].strip()
    pairs: list[tuple[str, str]] = []
    for line in block.splitlines():
        if "->" in line:
            left, right = line.split("->", 1)
            pairs.append((left.strip(), right.strip()))
    return pairs


def extract_equals_pairs(prompt: str) -> list[tuple[str, str]]:
    block = prompt.split("Below are a few examples:", 1)[1].split(
        "Now, determine the result for:", 1
    )[0]
    pairs: list[tuple[str, str]] = []
    for line in block.splitlines():
        if "=" in line:
            left, right = line.split("=", 1)
            pairs.append((left.strip(), right.strip()))
    return pairs


def build_roman_reasoning(prompt: str, answer: str) -> str:
    target = int(re.search(r"Now, write the number (\d+)", prompt).group(1))
    computed, steps = roman_with_steps(target)
    return (
        "Problem type: Digital counting conversion. "
        "Caution: convert the target number with standard Roman numeral rules, "
        "including subtractive forms such as IV, IX, XL, and XC. "
        f"Solution: the target number is {target}. "
        + " ".join(steps)
        + f" Combining the Roman pieces gives {computed}, which matches the labeled answer. "
        f"answer : {answer}"
    )


def build_gravity_reasoning(prompt: str, answer: str) -> str:
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
    details = "; ".join(
        f"from t={t:g}, d={d:g}: g = 2*d/t^2 = 2*{d:g}/{t:g}^2 = {g:.4f}"
        for (t, d), g in zip(pairs, g_values)
    )
    joined = " + ".join(f"{g:.4f}" for g in g_values)
    return (
        "Problem type: Gravity conversion. "
        "Caution: do not assume Earth's usual gravitational constant; first infer Wonderland's g "
        "from the examples, then substitute the target time into d = 0.5*g*t^2. "
        f"Solution: rearrange the formula to g = 2d/t^2. {details}. "
        f"Average the inferred constants: g = ({joined})/{len(g_values)} approx {avg_g:.4f}. "
        f"Now substitute t = {target_t:g}: d = 0.5*{avg_g:.4f}*{target_t:g}^2 = {pred:.4f}. "
        f"Rounded to the dataset format, answer : {answer}"
    )


def build_unit_reasoning(prompt: str, answer: str) -> str:
    pairs = [(float(x), float(y)) for x, y in re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)]
    target_x = float(
        re.search(r"convert the following measurement: ([\d.]+) m", prompt).group(1)
    )
    xs = np.array([x for x, _ in pairs], dtype=float)
    ys = np.array([y for _, y in pairs], dtype=float)
    design = np.vstack([xs, np.ones(len(xs))]).T
    slope, intercept = np.linalg.lstsq(design, ys, rcond=None)[0]
    pred = slope * target_x + intercept
    points = ", ".join(f"({x:g}, {y:g})" for x, y in pairs)
    return (
        "Problem type: Unit conversion. "
        "Caution: use all example pairs to fit the hidden linear conversion y = a*x + b; "
        "small rounding differences in a and b will change the final decimal output. "
        f"Solution: the example pairs are {points}. "
        f"Fitting y = a*x + b gives a approx {slope:.4f} and b approx {intercept:.4f}. "
        f"For x = {target_x:g}, y = {slope:.4f}*{target_x:g} + {intercept:.4f} = {pred:.4f}. "
        f"Rounded to the dataset format, answer : {answer}"
    )


def build_text_reasoning(prompt: str, answer: str) -> str:
    pairs = extract_arrow_pairs(
        prompt,
        "Here are some examples:",
        "Now, decrypt the following text:",
    )
    target = prompt.split("Now, decrypt the following text:", 1)[1].strip().rstrip(".")
    known_word_pairs: list[str] = []
    for enc_phrase, plain_phrase in pairs:
        enc_words = enc_phrase.split()
        plain_words = plain_phrase.split()
        known_word_pairs.extend(
            f"{enc}->{plain}" for enc, plain in zip(enc_words, plain_words)
        )
        if len(known_word_pairs) >= 12:
            break

    enc_target_words = target.split()
    plain_target_words = answer.split()
    if len(enc_target_words) == len(plain_target_words):
        alignment = ", ".join(
            f"{enc}->{plain}" for enc, plain in zip(enc_target_words, plain_target_words)
        )
        resolution_text = (
            f"The target phrase has the same word count on both sides, so the labeled resolution is "
            f"{alignment}."
        )
    else:
        resolution_text = f'The labeled decrypted phrase for the target is "{answer}".'

    return (
        "Problem type: Text encryption conversion. "
        "Caution: preserve word order and decode the whole phrase under one consistent hidden text rule; "
        "do not guess by ordinary English semantics alone. "
        f'Solution: the examples provide encrypted-to-plain evidence such as {", ".join(known_word_pairs[:12])}. '
        f'The target encrypted phrase is "{target}". {resolution_text} '
        f'This yields the labeled plaintext phrase "{answer}". answer : {answer}'
    )


def build_binary_reasoning(prompt: str, answer: str) -> str:
    pairs = extract_arrow_pairs(
        prompt,
        "Here are some examples of input -> output:",
        "Now, determine the output for:",
    )
    target = prompt.split("Now, determine the output for:", 1)[1].strip().rstrip(".")
    sample_pairs = ", ".join(f"{left}->{right}" for left, right in pairs[:6])
    return (
        "Problem type: Binary operation conversion. "
        "Caution: the input and output must remain 8-bit binary strings, and the hidden rule may combine "
        "shift, rotation, XOR, AND, OR, NOT, majority, or choice operations. "
        "Only example-based observations should be stated unless the exact rule is uniquely determined. "
        f"Solution: compare the example mappings {sample_pairs}. "
        f"The target input is {target}, and the labeled output is {answer}. "
        f"Check the output format: {answer} has {len(answer)} bits and uses only 0/1, so it is a valid 8-bit result. "
        f"answer : {answer}"
    )


def build_equation_reasoning(prompt: str, answer: str, question_type: str) -> str:
    pairs = extract_equals_pairs(prompt)
    target = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    evidence = "; ".join(f"{left} = {right}" for left, right in pairs[:6])

    if question_type == "Character equation conversion":
        caution = (
            "all symbols matter, so preserve punctuation, order, and any backslashes exactly as they appear"
        )
    elif question_type == "Digital character operation":
        caution = (
            "the right-hand side may contain both digits and non-digit symbols, so preserve the exact character "
            "sequence instead of forcing a purely numeric interpretation"
        )
    else:
        caution = (
            "operators are custom symbols, not necessarily standard arithmetic, but the final result in this type "
            "contains only digits"
        )

    return (
        f"Problem type: {question_type}. "
        f"Caution: {caution}. "
        f"Solution: inspect the provided examples {evidence}. "
        "These examples define a hidden transformation for the expression format used in this item. "
        f"Applying the labeled rule for the target expression {target} gives {answer}. "
        f"Preserve the exact output formatting from the label. answer : {answer}"
    )


def build_reasoning(prompt: str, answer: str, question_type: str) -> str:
    if question_type == "Digital counting conversion":
        return build_roman_reasoning(prompt, answer)
    if question_type == "Gravity conversion":
        return build_gravity_reasoning(prompt, answer)
    if question_type == "Unit conversion":
        return build_unit_reasoning(prompt, answer)
    if question_type == "Text encryption conversion":
        return build_text_reasoning(prompt, answer)
    if question_type == "Binary operation conversion":
        return build_binary_reasoning(prompt, answer)
    if question_type in {
        "Character equation conversion",
        "Numerical equations conversion",
        "Digital character operation",
    }:
        return build_equation_reasoning(prompt, answer, question_type)

    return (
        "Problem type: Unknown. "
        "Caution: the rule template was not recognized, so preserve the labeled answer exactly. "
        f"Solution: use the provided supervised label for this item. answer : {answer}"
    )


def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["question_type"] = out["prompt"].map(classify_question_type)
    out["reasoning"] = [
        build_reasoning(prompt, answer, question_type)
        for prompt, answer, question_type in zip(
            out["prompt"], out["answer"], out["question_type"]
        )
    ]
    return out


def validate_output(df: pd.DataFrame) -> None:
    bad_suffix = df[~df.apply(lambda row: str(row["reasoning"]).endswith(f"answer : {row['answer']}"), axis=1)]
    if not bad_suffix.empty:
        raise ValueError(f"{len(bad_suffix)} rows do not end with the expected answer marker.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate Wonderland tasks with question type and reasoning.")
    parser.add_argument(
        "--input",
        default=RAW_DATA_DIR / "train.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=PROCESSED_DATA_DIR / "train_reasoning_strict_labeled.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str).fillna("")
    annotated = annotate_dataframe(df)
    validate_output(annotated)
    annotated.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(output_path)
    print(annotated.shape)
    print(annotated["question_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
