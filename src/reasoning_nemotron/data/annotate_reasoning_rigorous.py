from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_nemotron.data.annotate_reasoning import classify_question_type, roman_with_steps
from reasoning_nemotron.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def sign_reverse(text: str) -> str:
    if text.startswith("-"):
        return "-" + text[1:][::-1]
    return text[::-1]


@dataclass(frozen=True)
class NumericCandidate:
    name: str
    func: object
    description: str


@dataclass(frozen=True)
class BinaryCandidate:
    name: str
    func: object
    description: str


@dataclass(frozen=True)
class CharTemplate:
    template: tuple[str, ...]
    description: str


ROMAN_TYPE = "Digital counting conversion"
GRAVITY_TYPE = "Gravity conversion"
UNIT_TYPE = "Unit conversion"
TEXT_TYPE = "Text encryption conversion"
BINARY_TYPE = "Binary operation conversion"
CHAR_EQ_TYPE = "Character equation conversion"
NUM_EQ_TYPE = "Numerical equations conversion"
DIGIT_CHAR_TYPE = "Digital character operation"


SOURCE_LABELS = {
    "a": "the first two-digit number as written",
    "ra": "the first number with its digits reversed",
    "b": "the second two-digit number as written",
    "rb": "the second number with its digits reversed",
}

OUTPUT_LABELS = {
    "id": "write the decimal result as-is",
    "rev": "reverse all characters in the decimal result",
    "signrev": "keep any leading minus sign in front and reverse only the digits",
    "neg": "prefix the decimal result with a minus sign",
    "negrev": "prefix a minus sign and reverse the digits of the magnitude",
}

TEMPLATE_LABELS = {
    "op": "the operator symbol",
    "l1": "left[0]",
    "l2": "left[1]",
    "r1": "right[0]",
    "r2": "right[1]",
    "left": "the full left pair",
    "left_rev": "the reversed left pair",
    "right": "the full right pair",
    "right_rev": "the reversed right pair",
    "common_left": "the left-pair symbols that also appear on the right",
    "common_right": "the right-pair symbols that also appear on the left",
    "left_only": "the left-pair symbols that are absent from the right",
    "right_only": "the right-pair symbols that are absent from the left",
    "all": "left followed by right",
    "all_rev": "the reverse of left followed by right",
}


def build_text_vocab(df: pd.DataFrame) -> dict[int, list[str]]:
    vocab: set[str] = set()
    text_df = df[df["prompt"].str.contains("secret encryption rules", regex=False)]
    for _, row in text_df.iterrows():
        block = row["prompt"].split("Here are some examples:", 1)[1].split(
            "Now, decrypt the following text:", 1
        )[0]
        for line in block.splitlines():
            if "->" in line:
                vocab.update(line.split("->", 1)[1].strip().split())
        vocab.update(str(row["answer"]).split())

    by_len: dict[int, list[str]] = {}
    for word in sorted(vocab):
        by_len.setdefault(len(word), []).append(word)
    return by_len


def build_numeric_candidates() -> list[NumericCandidate]:
    numeric_sources = {
        "a": lambda a, b: int(a),
        "ra": lambda a, b: int(a[::-1]),
        "b": lambda a, b: int(b),
        "rb": lambda a, b: int(b[::-1]),
    }
    string_sources = {
        "a": lambda a, b: a,
        "ra": lambda a, b: a[::-1],
        "b": lambda a, b: b,
        "rb": lambda a, b: b[::-1],
    }
    output_transforms = {
        "id": lambda s: s,
        "rev": lambda s: s[::-1],
        "signrev": sign_reverse,
        "neg": lambda s: s if s.startswith("-") else "-" + s,
        "negrev": lambda s: "-" + (s[1:] if s.startswith("-") else s)[::-1],
    }

    candidates: list[NumericCandidate] = []

    def arithmetic_desc(left_name: str, op_name: str, right_name: str, const: int, out_name: str) -> str:
        op_text = {
            "add": "add",
            "sub": "subtract the second from the first",
            "rsub": "subtract the first from the second",
            "absdiff": "take the absolute difference between",
            "prod": "multiply",
        }[op_name]
        if op_name == "absdiff":
            core = f"take {SOURCE_LABELS[left_name]} and {SOURCE_LABELS[right_name]}, then {op_text} them"
        else:
            core = f"{op_text} {SOURCE_LABELS[left_name]} and {SOURCE_LABELS[right_name]}"
        if const == 1:
            core += ", then add 1"
        elif const == -1:
            core += ", then subtract 1"
        core += f"; finally, {OUTPUT_LABELS[out_name]}"
        return core

    source_order = ["a", "b", "ra", "rb"]
    op_order = ["add", "sub", "rsub", "absdiff", "prod"]
    const_order = [0, -1, 1]
    out_order = ["id", "rev", "signrev", "neg", "negrev"]

    for left_name in source_order:
        for right_name in source_order:
            left_fn = numeric_sources[left_name]
            right_fn = numeric_sources[right_name]
            for op_name in op_order:
                for const in const_order:
                    for out_name in out_order:
                        out_fn = output_transforms[out_name]

                        def func(
                            a: str,
                            b: str,
                            left_fn=left_fn,
                            right_fn=right_fn,
                            op_name=op_name,
                            const=const,
                            out_fn=out_fn,
                        ) -> str:
                            x = left_fn(a, b)
                            y = right_fn(a, b)
                            if op_name == "add":
                                val = x + y
                            elif op_name == "sub":
                                val = x - y
                            elif op_name == "rsub":
                                val = y - x
                            elif op_name == "absdiff":
                                val = abs(x - y)
                            else:
                                val = x * y
                            return out_fn(str(val + const))

                        candidates.append(
                            NumericCandidate(
                                name=f"{left_name}_{op_name}_{right_name}_{const}_{out_name}",
                                func=func,
                                description=arithmetic_desc(left_name, op_name, right_name, const, out_name),
                            )
                        )

    concat_out_order = ["id", "rev"]
    for left_name in source_order:
        for right_name in source_order:
            left_fn = string_sources[left_name]
            right_fn = string_sources[right_name]
            for order_name in ["xy", "yx"]:
                for out_name in concat_out_order:
                    out_fn = output_transforms[out_name]
                    if order_name == "xy":
                        core = f"concatenate {SOURCE_LABELS[left_name]} followed by {SOURCE_LABELS[right_name]}"
                    else:
                        core = f"concatenate {SOURCE_LABELS[right_name]} followed by {SOURCE_LABELS[left_name]}"
                    description = f"{core}; finally, {OUTPUT_LABELS[out_name]}"

                    def func(
                        a: str,
                        b: str,
                        left_fn=left_fn,
                        right_fn=right_fn,
                        order_name=order_name,
                        out_fn=out_fn,
                    ) -> str:
                        if order_name == "xy":
                            text = left_fn(a, b) + right_fn(a, b)
                        else:
                            text = right_fn(a, b) + left_fn(a, b)
                        return out_fn(text)

                    candidates.append(
                        NumericCandidate(
                            name=f"{left_name}_concat_{order_name}_{right_name}_{out_name}",
                            func=func,
                            description=description,
                        )
                    )
    return candidates


def build_binary_candidates() -> list[BinaryCandidate]:
    def rol(x: int, n: int) -> int:
        return ((x << n) & 0xFF) | (x >> (8 - n))

    def ror(x: int, n: int) -> int:
        return (x >> n) | ((x << (8 - n)) & 0xFF)

    def shl(x: int, n: int) -> int:
        return (x << n) & 0xFF

    def shr(x: int, n: int) -> int:
        return x >> n

    def bit_not(x: int) -> int:
        return (~x) & 0xFF

    def majority(a: int, b: int, c: int) -> int:
        return (a & b) | (a & c) | (b & c)

    def choice(a: int, b: int, c: int) -> int:
        return (a & b) | (bit_not(a) & c)

    transforms: list[tuple[str, object, str]] = [
        ("x", lambda x: x, "the input bits"),
        ("notx", bit_not, "bitwise NOT of the input"),
    ]
    for n in [1, 2, 3, 4]:
        transforms.append((f"rol{n}", lambda x, n=n: rol(x, n), f"rotate left by {n}"))
        transforms.append((f"ror{n}", lambda x, n=n: ror(x, n), f"rotate right by {n}"))
    for n in [1, 2, 3]:
        transforms.append((f"shl{n}", lambda x, n=n: shl(x, n), f"shift left by {n} with zero fill"))
        transforms.append((f"shr{n}", lambda x, n=n: shr(x, n), f"shift right by {n} with zero fill"))

    candidates: list[BinaryCandidate] = []
    for name, func, desc in transforms:
        candidates.append(BinaryCandidate(name=name, func=func, description=desc))

    for (n1, f1, d1), (n2, f2, d2) in itertools.product(transforms, repeat=2):
        candidates.append(
            BinaryCandidate(
                name=f"{n1}_xor_{n2}",
                func=lambda x, f1=f1, f2=f2: f1(x) ^ f2(x),
                description=f"take {d1} XOR {d2}",
            )
        )
        candidates.append(
            BinaryCandidate(
                name=f"{n1}_and_{n2}",
                func=lambda x, f1=f1, f2=f2: f1(x) & f2(x),
                description=f"take {d1} AND {d2}",
            )
        )
        candidates.append(
            BinaryCandidate(
                name=f"{n1}_or_{n2}",
                func=lambda x, f1=f1, f2=f2: f1(x) | f2(x),
                description=f"take {d1} OR {d2}",
            )
        )

    for (n1, f1, d1), (n2, f2, d2), (n3, f3, d3) in itertools.product(transforms, repeat=3):
        candidates.append(
            BinaryCandidate(
                name=f"{n1}_xor_{n2}_xor_{n3}",
                func=lambda x, f1=f1, f2=f2, f3=f3: f1(x) ^ f2(x) ^ f3(x),
                description=f"XOR together {d1}, {d2}, and {d3}",
            )
        )
        candidates.append(
            BinaryCandidate(
                name=f"maj_{n1}_{n2}_{n3}",
                func=lambda x, f1=f1, f2=f2, f3=f3: majority(f1(x), f2(x), f3(x)),
                description=f"take the bitwise majority of {d1}, {d2}, and {d3}",
            )
        )
        candidates.append(
            BinaryCandidate(
                name=f"ch_{n1}_{n2}_{n3}",
                func=lambda x, f1=f1, f2=f2, f3=f3: choice(f1(x), f2(x), f3(x)),
                description=f"take the bitwise choice function with selector {d1}, true branch {d2}, and false branch {d3}",
            )
        )
    return candidates


def build_char_templates() -> list[CharTemplate]:
    atom_order = [
        "op",
        "l1",
        "l2",
        "r1",
        "r2",
        "left",
        "left_rev",
        "right",
        "right_rev",
        "common_left",
        "common_right",
        "left_only",
        "right_only",
        "all",
        "all_rev",
        "",
    ]
    templates: list[CharTemplate] = []
    for size in [1, 2, 3]:
        for combo in itertools.product(atom_order, repeat=size):
            if all(part == "" for part in combo):
                continue
            description = "concatenate " + " + ".join(TEMPLATE_LABELS.get(part, "''") for part in combo if part)
            templates.append(CharTemplate(template=combo, description=description))
    return templates


NUMERIC_CANDIDATES = build_numeric_candidates()
BINARY_CANDIDATES = build_binary_candidates()
CHAR_TEMPLATES = build_char_templates()


def parse_equation_expr(expr: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(..)(.)(..)$", expr)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def extract_equation_examples(prompt: str) -> list[tuple[str, str]]:
    block = prompt.split("Below are a few examples:", 1)[1].split(
        "Now, determine the result for:", 1
    )[0]
    examples: list[tuple[str, str]] = []
    for line in block.splitlines():
        if "=" in line:
            left, right = line.split("=", 1)
            examples.append((left.strip(), right.strip()))
    return examples


def build_text_maps(prompt: str) -> tuple[dict[str, str], dict[str, str]]:
    block = prompt.split("Here are some examples:", 1)[1].split(
        "Now, decrypt the following text:", 1
    )[0]
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for line in block.splitlines():
        if "->" not in line:
            continue
        encrypted, plain = line.split("->", 1)
        encrypted = encrypted.strip()
        plain = plain.strip()
        for enc_char, plain_char in zip(encrypted, plain):
            if enc_char == " ":
                continue
            forward[enc_char] = plain_char
            reverse[plain_char] = enc_char
    return forward, reverse


def text_word_matches(enc_word: str, candidate: str, forward: dict[str, str], reverse: dict[str, str]) -> bool:
    if len(enc_word) != len(candidate):
        return False
    local_forward = dict(forward)
    local_reverse = dict(reverse)
    for enc_char, plain_char in zip(enc_word, candidate):
        if enc_char in local_forward and local_forward[enc_char] != plain_char:
            return False
        if plain_char in local_reverse and local_reverse[plain_char] != enc_char:
            return False
        local_forward[enc_char] = plain_char
        local_reverse[plain_char] = enc_char
    return True


def decode_text_target(prompt: str, vocab_by_len: dict[int, list[str]]) -> tuple[list[str], dict[str, str], list[str]]:
    forward, reverse = build_text_maps(prompt)
    target = prompt.split("Now, decrypt the following text:", 1)[1].strip().rstrip(".")
    words = target.split()
    resolved: list[str | None] = [None] * len(words)
    pattern_notes: list[str] = []

    changed = True
    while changed:
        changed = False
        for idx, enc_word in enumerate(words):
            if resolved[idx] is not None:
                continue
            partial = "".join(forward.get(ch, "?") for ch in enc_word)
            if "?" not in partial:
                resolved[idx] = partial
                changed = True
                continue

            candidates = [
                word
                for word in vocab_by_len.get(len(enc_word), [])
                if text_word_matches(enc_word, word, forward, reverse)
            ]
            if len(candidates) == 1:
                word = candidates[0]
                resolved[idx] = word
                for enc_char, plain_char in zip(enc_word, word):
                    forward[enc_char] = plain_char
                    reverse[plain_char] = enc_char
                pattern_notes.append(f"{enc_word} -> {partial} -> {word}")
                changed = True

    final_words = [word or "".join(forward.get(ch, "?") for ch in enc_word) for word, enc_word in zip(resolved, words)]
    return final_words, forward, pattern_notes


def apply_char_template(expr: str, template: tuple[str, ...]) -> str:
    left1, left2, operator, right1, right2 = expr[0], expr[1], expr[2], expr[3], expr[4]
    left = left1 + left2
    right = right1 + right2

    def common(a: str, b: str) -> str:
        return "".join(ch for ch in a if ch in b)

    def diff(a: str, b: str) -> str:
        return "".join(ch for ch in a if ch not in b)

    pieces = {
        "": "",
        "op": operator,
        "l1": left1,
        "l2": left2,
        "r1": right1,
        "r2": right2,
        "left": left,
        "left_rev": left[::-1],
        "right": right,
        "right_rev": right[::-1],
        "common_left": common(left, right),
        "common_right": common(right, left),
        "left_only": diff(left, right),
        "right_only": diff(right, left),
        "all": left + right,
        "all_rev": (left + right)[::-1],
    }
    return "".join(pieces[part] for part in template)


def format_binary(value: int) -> str:
    return format(value & 0xFF, "08b")


def choose_binary_candidate(prompt: str, answer: str) -> BinaryCandidate | None:
    block = prompt.split("Here are some examples of input -> output:", 1)[1].split(
        "Now, determine the output for:", 1
    )[0]
    examples: list[tuple[int, int]] = []
    for line in block.splitlines():
        if "->" in line:
            left, right = line.split("->", 1)
            examples.append((int(left.strip(), 2), int(right.strip(), 2)))
    target = int(prompt.split("Now, determine the output for:", 1)[1].strip().rstrip("."), 2)

    for candidate in BINARY_CANDIDATES:
        if all(candidate.func(left) == right for left, right in examples):
            if format_binary(candidate.func(target)) == answer:
                return candidate
    return None


def choose_numeric_candidate(prompt: str, answer: str, allow_wrappers: bool) -> tuple[NumericCandidate, str] | None:
    target_text = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    target = parse_equation_expr(target_text)
    if not target:
        return None
    target_left, target_op, target_right = target
    examples = []
    for expr, result in extract_equation_examples(prompt):
        parsed = parse_equation_expr(expr)
        if parsed and parsed[1] == target_op:
            examples.append((parsed[0], parsed[2], result))
    if not examples:
        return None

    wrappers = [
        ("plain", lambda op, text: text, "write that result directly"),
        ("prefix", lambda op, text: op + text, "prefix the operator symbol to that result"),
        ("suffix", lambda op, text: text + op, "append the operator symbol to that result"),
    ]
    selected_wrappers = wrappers if allow_wrappers else wrappers[:1]

    for candidate in NUMERIC_CANDIDATES:
        for wrapper_name, wrapper_fn, wrapper_desc in selected_wrappers:
            if all(wrapper_fn(target_op, candidate.func(left, right)) == result for left, right, result in examples):
                if wrapper_fn(target_op, candidate.func(target_left, target_right)) == answer:
                    wrapped_description = candidate.description
                    if wrapper_name != "plain":
                        wrapped_description += f"; then {wrapper_desc}"
                    return candidate, wrapped_description
    return None


def choose_char_template(prompt: str, answer: str) -> CharTemplate | None:
    target_text = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    target = parse_equation_expr(target_text)
    if not target:
        return None
    target_op = target[1]
    examples = []
    for expr, result in extract_equation_examples(prompt):
        parsed = parse_equation_expr(expr)
        if parsed and parsed[1] == target_op:
            examples.append((expr, result))
    if not examples:
        return None

    for template in CHAR_TEMPLATES:
        if all(apply_char_template(expr, template.template) == result for expr, result in examples):
            if apply_char_template(target_text, template.template) == answer:
                return template
    return None


def build_roman_reasoning(prompt: str, answer: str) -> str:
    target = int(re.search(r"Now, write the number (\d+)", prompt).group(1))
    computed, steps = roman_with_steps(target)
    return (
        f"Problem type: {ROMAN_TYPE}. "
        "Caution: apply standard Roman numeral notation, including subtractive forms such as IV, IX, XL, and XC. "
        f"Solution: the target number is {target}. "
        + " ".join(steps)
        + f" Combining the pieces gives {computed}. answer : {answer}"
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
        f"from t={t:g}, d={d:g}: g = 2*{d:g}/{t:g}^2 = {g:.4f}" for (t, d), g in zip(pairs, g_values)
    )
    return (
        f"Problem type: {GRAVITY_TYPE}. "
        "Caution: first infer Wonderland's gravitational constant from the examples, then plug the target time into d = 0.5*g*t^2. "
        f"Solution: {details}. "
        f"The average inferred constant is g = {avg_g:.4f}. "
        f"For t = {target_t:g}, d = 0.5*{avg_g:.4f}*{target_t:g}^2 = {pred:.4f}. "
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
    return (
        f"Problem type: {UNIT_TYPE}. "
        "Caution: fit the hidden linear conversion y = a*x + b from all examples before converting the target value. "
        f"Solution: least-squares fitting on the example pairs gives a = {slope:.4f} and b = {intercept:.4f}. "
        f"For x = {target_x:g}, y = {slope:.4f}*{target_x:g} + {intercept:.4f} = {pred:.4f}. "
        f"Rounded to the dataset format, answer : {answer}"
    )


def build_text_reasoning(prompt: str, answer: str, vocab_by_len: dict[int, list[str]]) -> str:
    decoded_words, forward, pattern_notes = decode_text_target(prompt, vocab_by_len)
    target = prompt.split("Now, decrypt the following text:", 1)[1].strip().rstrip(".")
    used_letters = []
    for char in target:
        if char != " " and char in forward and char not in used_letters:
            used_letters.append(char)
    mapping_text = ", ".join(f"{char}->{forward[char]}" for char in used_letters[:14])
    resolved_target = " ".join(decoded_words)
    note_text = ""
    if pattern_notes:
        note_text = " Partial word patterns that become uniquely determined under the same substitution include " + "; ".join(pattern_notes[:4]) + "."
    return (
        f"Problem type: {TEXT_TYPE}. "
        "Caution: this is a monoalphabetic substitution, so the same encrypted letter must decode to the same plaintext letter everywhere in the item. "
        f"Solution: build the letter mapping from the examples. For the letters needed in the target, the examples give {mapping_text}.{note_text} "
        f"Applying that substitution to \"{target}\" yields \"{resolved_target}\". answer : {answer}"
    )


def build_binary_reasoning(prompt: str, answer: str) -> str:
    candidate = choose_binary_candidate(prompt, answer)
    target_text = prompt.split("Now, determine the output for:", 1)[1].strip().rstrip(".")
    if candidate is None:
        return (
            f"Problem type: {BINARY_TYPE}. "
            "Caution: the prompt allows many bitwise formulas involving shifts, rotations, XOR, AND, OR, NOT, majority, and choice. "
            "For this item, the displayed examples do not isolate one unique rule strongly enough to justify inventing a specific formula. "
            f"To stay rigorous, do not fabricate an unsupported derivation. The supervised gold output for target {target_text} is {answer}. "
            f"answer : {answer}"
        )

    block = prompt.split("Here are some examples of input -> output:", 1)[1].split(
        "Now, determine the output for:", 1
    )[0]
    examples = []
    for line in block.splitlines():
        if "->" in line:
            left, right = line.split("->", 1)
            left_text = left.strip()
            examples.append((left_text, right.strip(), format_binary(candidate.func(int(left_text, 2)))))
    target_output = format_binary(candidate.func(int(target_text, 2)))
    sample_checks = "; ".join(
        f"{left} -> {calc}" for left, _, calc in examples[:3]
    )
    return (
        f"Problem type: {BINARY_TYPE}. "
        "Caution: keep every intermediate and final value as an 8-bit binary string. "
        f"Solution: a simple rule consistent with the displayed examples is to {candidate.description}. "
        f"Checking the examples gives {sample_checks}. "
        f"Applying the same rule to {target_text} gives {target_output}. answer : {answer}"
    )


def build_numeric_equation_reasoning(prompt: str, answer: str, question_type: str) -> str:
    candidate_info = choose_numeric_candidate(
        prompt,
        answer,
        allow_wrappers=(question_type == DIGIT_CHAR_TYPE),
    )
    target_text = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    target = parse_equation_expr(target_text)
    assert target is not None
    target_left, target_op, target_right = target

    if candidate_info is None:
        return (
            f"Problem type: {question_type}. "
            "Caution: this item uses operator-specific hidden rules, and the available same-operator examples are not sufficient to pin down one defensible formula for the target. "
            f"To stay rigorous, do not pretend to derive a rule that the prompt does not determine. The supervised gold result for {target_text} is {answer}. "
            f"answer : {answer}"
        )

    candidate, description = candidate_info
    example_checks = []
    for expr, result in extract_equation_examples(prompt):
        parsed = parse_equation_expr(expr)
        if parsed and parsed[1] == target_op:
            calc = candidate.func(parsed[0], parsed[2])
            if question_type == DIGIT_CHAR_TYPE:
                if result == parsed[1] + calc:
                    calc = parsed[1] + calc
                elif result == calc + parsed[1]:
                    calc = calc + parsed[1]
            example_checks.append(f"{expr} -> {calc}")
            if len(example_checks) == 3:
                break

    target_calc = candidate.func(target_left, target_right)
    if question_type == DIGIT_CHAR_TYPE:
        if answer == target_op + target_calc:
            target_calc = target_op + target_calc
        elif answer == target_calc + target_op:
            target_calc = target_calc + target_op

    return (
        f"Problem type: {question_type}. "
        f"Caution: the operator is custom, so infer its behavior only from the same-operator examples in this item. "
        f"Solution: a simple rule consistent with all displayed `{target_op}` examples is to {description}. "
        f"That rule checks out on examples such as {'; '.join(example_checks)}. "
        f"Applying it to {target_text} gives {target_calc}. answer : {answer}"
    )


def build_char_equation_reasoning(prompt: str, answer: str) -> str:
    template = choose_char_template(prompt, answer)
    target_text = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    target = parse_equation_expr(target_text)
    assert target is not None
    target_op = target[1]

    if template is None:
        return (
            f"Problem type: {CHAR_EQ_TYPE}. "
            "Caution: the symbol rule here is highly item-specific, and the available same-operator evidence does not determine a trustworthy string-construction rule for the target. "
            f"To avoid inventing a false derivation, keep the reasoning honest: the supervised gold result for {target_text} is {answer}. "
            f"answer : {answer}"
        )

    example_checks = []
    for expr, result in extract_equation_examples(prompt):
        parsed = parse_equation_expr(expr)
        if parsed and parsed[1] == target_op:
            example_checks.append(f"{expr} -> {apply_char_template(expr, template.template)}")
            if len(example_checks) == 3:
                break
    target_calc = apply_char_template(target_text, template.template)
    return (
        f"Problem type: {CHAR_EQ_TYPE}. "
        "Caution: preserve every symbol exactly; order and punctuation are the whole problem. "
        f"Solution: a simple same-operator string rule consistent with the examples is to {template.description}. "
        f"This reproduces examples such as {'; '.join(example_checks)}. "
        f"Applying it to {target_text} gives {target_calc}. answer : {answer}"
    )


def build_reasoning(row: pd.Series, vocab_by_len: dict[int, list[str]]) -> tuple[str, str]:
    prompt = row["prompt"]
    answer = str(row["answer"])
    question_type = row["question_type"]

    if question_type == ROMAN_TYPE:
        return build_roman_reasoning(prompt, answer), "derived"
    if question_type == GRAVITY_TYPE:
        return build_gravity_reasoning(prompt, answer), "derived"
    if question_type == UNIT_TYPE:
        return build_unit_reasoning(prompt, answer), "derived"
    if question_type == TEXT_TYPE:
        return build_text_reasoning(prompt, answer, vocab_by_len), "derived"
    if question_type == BINARY_TYPE:
        reasoning = build_binary_reasoning(prompt, answer)
        return reasoning, "derived" if "simple rule consistent" in reasoning else "underdetermined"
    if question_type in {NUM_EQ_TYPE, DIGIT_CHAR_TYPE}:
        reasoning = build_numeric_equation_reasoning(prompt, answer, question_type)
        return reasoning, "derived" if "simple rule consistent" in reasoning else "underdetermined"
    if question_type == CHAR_EQ_TYPE:
        reasoning = build_char_equation_reasoning(prompt, answer)
        return reasoning, "derived" if "simple same-operator string rule" in reasoning else "underdetermined"

    reasoning = (
        "Problem type: Unknown. Caution: the template was not recognized, so avoid fabricating a derivation. "
        f"The supervised gold answer is {answer}. answer : {answer}"
    )
    return reasoning, "underdetermined"


def annotate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["question_type"] = out["prompt"].map(classify_question_type)
    vocab_by_len = build_text_vocab(out)

    reasonings: list[str] = []
    statuses: list[str] = []
    for _, row in out.iterrows():
        reasoning, status = build_reasoning(row, vocab_by_len)
        reasonings.append(reasoning)
        statuses.append(status)
    out["reasoning"] = reasonings

    audit = out[["id", "question_type"]].copy()
    audit["rewrite_status"] = statuses
    return out, audit


def validate(df: pd.DataFrame) -> None:
    bad = df[~df.apply(lambda row: str(row["reasoning"]).endswith(f"answer : {row['answer']}"), axis=1)]
    if not bad.empty:
        raise ValueError(f"{len(bad)} rows do not end with 'answer : <answer>'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate more rigorous reasoning annotations.")
    parser.add_argument(
        "--input",
        default=RAW_DATA_DIR / "train.csv",
    )
    parser.add_argument(
        "--output",
        default=PROCESSED_DATA_DIR / "train_reasoning_rigorous.csv",
    )
    parser.add_argument(
        "--audit-output",
        default=PROCESSED_DATA_DIR / "train_reasoning_rigorous_audit.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input), dtype=str).fillna("")
    annotated, audit = annotate(df)
    validate(annotated)

    annotated.to_csv(Path(args.output), index=False, encoding="utf-8-sig")
    audit.to_csv(Path(args.audit_output), index=False, encoding="utf-8-sig")

    print(Path(args.output))
    print(annotated.shape)
    print(annotated["question_type"].value_counts().to_string())
    print(audit["rewrite_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
