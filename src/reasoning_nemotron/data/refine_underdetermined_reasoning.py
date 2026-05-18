from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd

from reasoning_nemotron.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, TEMP_DIR


BIT_TYPE = "Binary operation conversion"
CHAR_TYPE = "Character equation conversion"
NUM_TYPE = "Numerical equations conversion"
DIGIT_CHAR_TYPE = "Digital character operation"


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""


def extract_investigation_predicted(text: str) -> str:
    match = re.search(
        r"predicted answer:\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def extract_block(text: str, header: str, next_headers: list[str]) -> str:
    next_part = "|".join(re.escape(h) for h in next_headers)
    pattern = rf"^{re.escape(header)}:\s*\n(.*?)(?=^(?:{next_part}):|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def cleaned_block_lines(block: str) -> list[str]:
    lines: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        line = re.sub(r"^[\-\*\d\.]+\s*", "", line)
        if line:
            lines.append(line)
    return lines


def load_ref_categories(ref_root: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    with (ref_root / "problems.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            categories[obj["id"]] = obj["category"]
    return categories


def load_bit_solver(ref_root: Path):
    path = ref_root / "investigators" / "bit_manipulation.py"
    module = types.ModuleType("bit_solver_module")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module.solve_problem


def load_numeric_reasoner(ref_root: Path):
    if str(ref_root) not in sys.path:
        sys.path.append(str(ref_root))
    eq_module = importlib.import_module("reasoners.equation_numeric")
    store_module = importlib.import_module("reasoners.store_types")
    return eq_module, store_module


def parse_bit_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    block = prompt.split("Here are some examples of input -> output:", 1)[1].split(
        "Now, determine the output for:", 1
    )[0]
    examples: list[tuple[str, str]] = []
    for line in block.splitlines():
        if "->" in line:
            left, right = line.split("->", 1)
            examples.append((left.strip(), right.strip()))
    query = prompt.split("Now, determine the output for:", 1)[1].strip().rstrip(".")
    return examples, query


def parse_equation_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    block = prompt.split("Below are a few examples:", 1)[1].split(
        "Now, determine the result for:", 1
    )[0]
    examples: list[tuple[str, str]] = []
    for line in block.splitlines():
        if "=" in line:
            left, right = line.split("=", 1)
            examples.append((left.strip(), right.strip()))
    query = prompt.split("Now, determine the result for:", 1)[1].strip().rstrip(".")
    return examples, query


def parse_symbol_digit_map(block: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in cleaned_block_lines(block):
        match = re.match(r"`(.+)`\s*=\s*(\d+)$", line)
        if match:
            mapping[match.group(1)] = match.group(2)
    return mapping


def parse_operator_map(block: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in cleaned_block_lines(block):
        match = re.match(r"`(.+)`\s*=\s*(.+)$", line)
        if match:
            mapping[match.group(1)] = match.group(2)
    return mapping


def build_bit_formula_reasoning(query: str, answer: str, rule: str) -> str:
    return (
        f"Problem type: {BIT_TYPE}. "
        "Caution: this item needs the broader corpus search over rotations, shifts, NOT, and nested XOR/AND/OR compositions. "
        f"Solution: the verified rule that reproduces every shown example is `{rule}`. "
        f"Applying that rule to {query} gives {answer}. answer : {answer}"
    )


def build_bit_selected_reasoning(query: str, answer: str, selected_lines: list[str]) -> str:
    summary = "; ".join(selected_lines[:8])
    return (
        f"Problem type: {BIT_TYPE}. "
        "Caution: this item is better explained by per-bit dependency matching than by one simple global shift formula. "
        f"Solution: the verified bit trace selects `{summary}`. "
        f"Applying those selected per-bit rules to {query} yields {answer}. answer : {answer}"
    )


def build_numeric_investigation_reasoning(
    question_type: str,
    answer: str,
    ref_category: str,
    text: str,
) -> str:
    rule_block = extract_block(
        text,
        "inferred rule",
        ["why this fits the examples", "step-by-step application to the query", "confidence note", "predicted answer"],
    )
    step_block = extract_block(
        text,
        "step-by-step application to the query",
        ["predicted answer", "confidence note"],
    )
    rule_lines = cleaned_block_lines(rule_block)
    step_lines = cleaned_block_lines(step_block)
    caution = (
        "this remains a guess-type operator puzzle, so only use the rule that the broader corpus investigation can defend cleanly."
        if ref_category.endswith("guess")
        else "this row needed the broader corpus investigation to disambiguate the operator behavior."
    )
    solution_parts = []
    if rule_lines:
        solution_parts.append("Rule: " + " ".join(rule_lines[:4]))
    if step_lines:
        solution_parts.append("Query application: " + " ".join(step_lines[:5]))
    solution_text = " ".join(solution_parts) if solution_parts else "The broader corpus investigation reproduces the gold answer."
    return (
        f"Problem type: {question_type}. "
        f"Caution: {caution} "
        f"Solution: {solution_text} answer : {answer}"
    )


def build_cryptarithm_investigation_reasoning(answer: str, text: str) -> str:
    rule_block = extract_block(
        text,
        "inferred rule",
        ["why this fits the examples", "step-by-step application to the query", "confidence note", "predicted answer"],
    )
    step_block = extract_block(
        text,
        "step-by-step application to the query",
        ["predicted answer", "confidence note"],
    )
    symbol_block = extract_block(
        text,
        "symbol-to-digit mapping",
        ["operator-to-operation mapping", "examples", "query", "predicted answer"],
    )
    op_block = extract_block(
        text,
        "operator-to-operation mapping",
        ["examples", "query", "predicted answer"],
    )

    symbol_map = parse_symbol_digit_map(symbol_block)
    op_map = parse_operator_map(op_block)
    symbol_items = [f"{k}={v}" for k, v in list(symbol_map.items())[:6]]
    op_items = [f"{k}->{v}" for k, v in list(op_map.items())[:4]]

    pieces = []
    rule_lines = cleaned_block_lines(rule_block)
    if rule_lines:
        pieces.append(" ".join(rule_lines[:4]))
    if symbol_items:
        pieces.append("One consistent symbol map is " + ", ".join(symbol_items) + ".")
    if op_items:
        pieces.append("Operator meanings include " + ", ".join(op_items) + ".")
    step_lines = cleaned_block_lines(step_block)
    if step_lines:
        pieces.append(" ".join(step_lines[:5]))

    solution_text = " ".join(pieces) if pieces else "A verified corpus-level cryptarithm assignment reproduces the gold answer."
    return (
        f"Problem type: {CHAR_TYPE}. "
        "Caution: this row becomes solvable only after the broader corpus cryptarithm analysis. "
        f"Solution: {solution_text} answer : {answer}"
    )


def build_cryptarithm_concat_reasoning(query: str, answer: str) -> str:
    left = query[:2]
    right = query[3:]
    if answer == left + right:
        operation = "concatenate the left pair followed by the right pair"
    else:
        operation = "concatenate the right pair followed by the left pair"
    return (
        f"Problem type: {CHAR_TYPE}. "
        "Caution: external corpus analysis tags this as a guess-type symbol equation, and the only verified fallback here is a concatenation pattern. "
        f"Solution: {operation}. Applying that to {query} gives {answer}. answer : {answer}"
    )


def build_numeric_corpus_reasoning(
    question_type: str,
    answer: str,
    q_op: str | None,
    effective_q_op: str | None,
    found_op,
    steps: list[str],
    ref_category: str,
) -> str:
    if effective_q_op == q_op:
        operator_text = f"the `{q_op}` examples"
    else:
        operator_text = (
            f"the target operator `{q_op}` is unseen locally, so the broader corpus fallback uses absolute difference while preserving the detected formatting convention from `{effective_q_op}`"
        )
    caution = (
        "this is a guess-type operator puzzle, so keep only the broader rule family that actually reproduces the gold answer."
        if ref_category.endswith("guess")
        else "this row needed the broader arithmetic and digit-wise search used in the external corpus."
    )
    op_desc = found_op.op_name
    step_text = " ".join(steps[:4])
    return (
        f"Problem type: {question_type}. "
        f"Caution: {caution} "
        f"Solution: the matching rule is found from {operator_text}; core operation = {op_desc}. "
        f"{step_text} answer : {answer}"
    )


def build_unresolved_reasoning(question_type: str, answer: str, ref_category: str) -> str:
    if ref_category.endswith("guess"):
        caution = (
            f"external corpus analysis classifies this item as `{ref_category}`, so the examples do not uniquely determine one reliable rule."
        )
    else:
        caution = (
            f"external corpus analysis classifies this item as `{ref_category}`, but none of the verified local or borrowed solvers reproduced the gold answer cleanly."
        )
    return (
        f"Problem type: {question_type}. "
        f"Caution: {caution} "
        f"To avoid fabricating a derivation, keep this row unresolved. answer : {answer}"
    )


def solve_numeric_with_external(eq_module, prompt: str, answer: str):
    examples, query = parse_equation_prompt(prompt)
    expr_re = re.compile(r"^(\d+)(\D)(\d+)$")
    parsed: list[tuple[str, str, str, str]] = []
    for left, right in examples:
        match = expr_re.fullmatch(left)
        if match:
            parsed.append((match.group(1), match.group(2), match.group(3), right))

    by_op: dict[str, list[tuple[str, str, str]]] = {}
    for a, op, b, out in parsed:
        by_op.setdefault(op, []).append((a, b, out))

    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    for op_char, group in by_op.items():
        any_neg_suffixed = op_char != "-" and any(
            out.endswith("-") and len(out) > 1 for _, _, out in group
        )
        any_neg_prefixed = op_char != "-" and any(
            out.startswith("-") and len(out) > 1 for _, _, out in group
        )
        any_suffixed = any(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
        any_prefixed = any(out.startswith(op_char) and len(out) > 1 for _, _, out in group)

        fmt = "num"
        transformed = list(group)
        if any_neg_suffixed:
            fmt = "neg_suffix"
            transformed = [
                (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
                for a, b, out in group
            ]
        elif any_neg_prefixed:
            fmt = "neg_prefix"
        elif any_suffixed:
            fmt = "neg_suffix"
            transformed = [
                (
                    a,
                    b,
                    "-" + out[: -len(op_char)] if out.endswith(op_char) and len(out) > 1 else out,
                )
                for a, b, out in group
            ]
        elif any_prefixed:
            fmt = "neg_prefix"
            transformed = [
                (
                    a,
                    b,
                    "-" + out[len(op_char) :] if out.startswith(op_char) and len(out) > 1 else out,
                )
                for a, b, out in group
            ]

        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = transformed

    q_match = expr_re.fullmatch(query)
    if not q_match:
        return None
    q_op = q_match.group(2)
    effective_q_op = q_op
    if q_op not in by_op and by_op:
        effective_q_op = max(by_op, key=lambda op: len(by_op[op]))

    found_ops = {}
    for op_char, group in sorted(by_op.items()):
        if effective_q_op is not None and op_char != effective_q_op and len(by_op) > 1:
            continue

        detected_fmt = detected_fmts[op_char]
        group = transformed_groups[op_char]
        found = None
        candidate_sets = [
            eq_module._common_candidates,
            eq_module._rare_candidates,
        ]

        for cand_fn in candidate_sets:
            for rev_ops, rev_res in (
                (True, True),
                (False, False),
                (True, False),
                (False, True),
            ):
                first_a, first_b, _ = group[0]
                probe_a = first_a[::-1] if rev_ops else first_a
                probe_b = first_b[::-1] if rev_ops else first_b
                for cand_name, _ in cand_fn(int(probe_a), int(probe_b), probe_a, probe_b):
                    all_pass = True
                    for ex_a, ex_b, expected in group:
                        real_a = ex_a[::-1] if rev_ops else ex_a
                        real_b = ex_b[::-1] if rev_ops else ex_b
                        raw = next(
                            result
                            for name, result in eq_module._all_candidates(
                                int(real_a), int(real_b), real_a, real_b
                            )
                            if name == cand_name
                        )
                        final = eq_module._rev(raw) if rev_res else raw
                        if final != expected:
                            all_pass = False
                            break
                    if all_pass:
                        found = eq_module.FoundOp(
                            op_name=cand_name,
                            rev_ops=rev_ops,
                            rev_res=rev_res,
                            fmt=detected_fmt,
                            op_char=op_char,
                        )
                        break
                if found:
                    break
            if found:
                break

        if found:
            found_ops[op_char] = found

    if effective_q_op not in found_ops:
        return None

    left = q_match.group(1)
    right = q_match.group(3)
    if effective_q_op != q_op:
        found_op = eq_module.FoundOp(
            op_name="absolute difference",
            rev_ops=False,
            rev_res=False,
            fmt=found_ops[effective_q_op].fmt,
            op_char=q_op or "",
        )
    else:
        found_op = found_ops[effective_q_op]

    result, steps = eq_module._apply_op(found_op, left, right)
    if result != answer:
        return None
    return {
        "query": query,
        "q_op": q_op,
        "effective_q_op": effective_q_op,
        "found_op": found_op,
        "steps": steps,
    }


def refine_rows(
    base_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    train_df: pd.DataFrame,
    ref_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_categories = load_ref_categories(ref_root)
    bit_solver = load_bit_solver(ref_root)
    eq_module, _ = load_numeric_reasoner(ref_root)

    refined = base_df.copy()
    audit = audit_df.copy()
    audit = audit.rename(columns={"rewrite_status": "prev_rewrite_status"})
    refined = refined.merge(audit, on=["id", "question_type"], how="left")
    refined["ref_category"] = refined["id"].map(ref_categories)
    refined["refine_method"] = ""
    refined["final_rewrite_status"] = refined["prev_rewrite_status"].fillna("derived")

    reason_dir = ref_root / "reasoning"
    inv_dir = ref_root / "investigations"
    answer_map = dict(zip(train_df["id"], train_df["answer"]))

    for idx, row in refined.iterrows():
        if row["prev_rewrite_status"] != "underdetermined":
            refined.at[idx, "final_rewrite_status"] = "derived_base"
            continue

        problem_id = row["id"]
        question_type = row["question_type"]
        answer = str(row["answer"])
        prompt = row["prompt"]
        ref_category = str(row["ref_category"])
        updated_reasoning = None
        method = None

        inv_path = inv_dir / f"{problem_id}.txt"
        if inv_path.exists():
            inv_text = inv_path.read_text(encoding="utf-8", errors="ignore")
            if extract_investigation_predicted(inv_text) == answer:
                if ref_category == "bit_manipulation":
                    rule_match = re.search(r"^rule:\s*(.+)$", inv_text, flags=re.IGNORECASE | re.MULTILINE)
                    if rule_match:
                        _, query = parse_bit_prompt(prompt)
                        updated_reasoning = build_bit_formula_reasoning(query, answer, rule_match.group(1).strip())
                    else:
                        updated_reasoning = build_numeric_investigation_reasoning(question_type, answer, ref_category, inv_text)
                elif ref_category.startswith("equation_numeric"):
                    updated_reasoning = build_numeric_investigation_reasoning(question_type, answer, ref_category, inv_text)
                elif ref_category.startswith("cryptarithm"):
                    updated_reasoning = build_cryptarithm_investigation_reasoning(answer, inv_text)
                if updated_reasoning:
                    method = "investigation"

        if updated_reasoning is None and ref_category == "bit_manipulation":
            examples, query = parse_bit_prompt(prompt)
            pred, rule, _ = bit_solver(
                {
                    "examples": [{"input_value": left, "output_value": right} for left, right in examples],
                    "question": query,
                    "answer": answer,
                }
            )
            if pred == answer:
                updated_reasoning = build_bit_formula_reasoning(query, answer, rule)
                method = "bit_solver"

        if updated_reasoning is None and ref_category == "bit_manipulation":
            reasoning_path = reason_dir / f"{problem_id}.txt"
            if reasoning_path.exists():
                text = reasoning_path.read_text(encoding="utf-8", errors="ignore")
                if extract_boxed(text) == answer:
                    selected_match = re.search(
                        r"^Selected\s*\n(.*?)\n\s*Applying to ",
                        text,
                        flags=re.MULTILINE | re.DOTALL,
                    )
                    if selected_match:
                        selected_lines = [
                            line.strip()
                            for line in selected_match.group(1).splitlines()
                            if line.strip()
                        ]
                        _, query = parse_bit_prompt(prompt)
                        updated_reasoning = build_bit_selected_reasoning(query, answer, selected_lines)
                        method = "bit_reasoning_trace"

        if updated_reasoning is None and ref_category.startswith("equation_numeric"):
            solved = solve_numeric_with_external(eq_module, prompt, answer)
            if solved is not None:
                updated_reasoning = build_numeric_corpus_reasoning(
                    question_type,
                    answer,
                    solved["q_op"],
                    solved["effective_q_op"],
                    solved["found_op"],
                    solved["steps"],
                    ref_category,
                )
                method = "numeric_solver"

        if updated_reasoning is None and ref_category == "cryptarithm_guess":
            _, query = parse_equation_prompt(prompt)
            left = query[:2]
            right = query[3:]
            if answer in {left + right, right + left}:
                updated_reasoning = build_cryptarithm_concat_reasoning(query, answer)
                method = "cryptarithm_concat_fallback"

        if updated_reasoning is None:
            updated_reasoning = build_unresolved_reasoning(question_type, answer, ref_category)
            method = "unresolved"
            final_status = "unresolved_guess" if ref_category.endswith("guess") else "unresolved_deduce"
        else:
            final_status = "derived_corpus"

        refined.at[idx, "reasoning"] = updated_reasoning
        refined.at[idx, "refine_method"] = method
        refined.at[idx, "final_rewrite_status"] = final_status

    output_df = refined.drop(columns=["prev_rewrite_status"]).copy()
    audit_out = refined[
        [
            "id",
            "question_type",
            "ref_category",
            "final_rewrite_status",
            "refine_method",
        ]
    ].copy()
    return output_df, audit_out


def validate(df: pd.DataFrame) -> None:
    bad = df[
        ~df.apply(
            lambda row: str(row["reasoning"]).endswith(f"answer : {row['answer']}"),
            axis=1,
        )
    ]
    if not bad.empty:
        raise ValueError(f"{len(bad)} rows do not end with 'answer : <answer>'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine underdetermined reasoning rows with broader corpus analysis.")
    parser.add_argument(
        "--train",
        default=RAW_DATA_DIR / "train.csv",
    )
    parser.add_argument(
        "--base",
        default=PROCESSED_DATA_DIR / "train_reasoning_rigorous.csv",
    )
    parser.add_argument(
        "--audit",
        default=PROCESSED_DATA_DIR / "train_reasoning_rigorous_audit.csv",
    )
    parser.add_argument(
        "--ref-root",
        default=TEMP_DIR / "nemotron_ref",
    )
    parser.add_argument(
        "--output",
        default=PROCESSED_DATA_DIR / "train_reasoning_refined.csv",
    )
    parser.add_argument(
        "--audit-output",
        default=PROCESSED_DATA_DIR / "train_reasoning_refined_audit.csv",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(Path(args.train), dtype=str).fillna("")
    base_df = pd.read_csv(Path(args.base), dtype=str).fillna("")
    audit_df = pd.read_csv(Path(args.audit), dtype=str).fillna("")

    refined_df, audit_out = refine_rows(base_df, audit_df, train_df, Path(args.ref_root))
    validate(refined_df)

    refined_df.to_csv(Path(args.output), index=False, encoding="utf-8-sig")
    audit_out.to_csv(Path(args.audit_output), index=False, encoding="utf-8-sig")

    print(Path(args.output))
    print(refined_df["final_rewrite_status"].value_counts().to_string())
    print(audit_out["refine_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
