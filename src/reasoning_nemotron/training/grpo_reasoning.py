from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reasoning_nemotron.evaluation.local_metric import extract_boxed_answer, verify
from reasoning_nemotron.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ensure_dir


SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Infer the hidden rule carefully, keep the reasoning concise but explicit, "
    "and end with the final answer in \\boxed{}."
)


@dataclass
class ReasoningGRPOProjectConfig:
    base_model_path: str = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    sft_adapter_path: str | None = None
    train_file: str = str(PROCESSED_DATA_DIR / "train_single_item_strict_train_refined.csv")
    output_dir: str = str(ARTIFACTS_DIR / "checkpoints" / "grpo_reasoning")
    learning_rate: float = 5e-6
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 768
    max_completion_length: int = 256
    beta: float = 0.04
    answer_reward_weight: float = 0.7
    format_reward_weight: float = 0.1
    question_type_reward_weight: float = 0.1
    consistency_reward_weight: float = 0.1
    excessive_length_penalty: float = 0.05


def load_config(config_path: str | None) -> ReasoningGRPOProjectConfig:
    config = ReasoningGRPOProjectConfig()
    if not config_path:
        return config

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key, value in payload.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown config field: {key}")
        setattr(config, key, value)
    return config


def build_prompt(raw_prompt: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Problem:\n{raw_prompt}\n\n"
        "Respond with reasoning first and finish with one final \\boxed{} answer."
    )


def load_project_dataset(train_file: Path) -> Dataset:
    frame = pd.read_csv(train_file, dtype=str).fillna("")
    required = {"prompt", "answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GRPO train file is missing columns: {sorted(missing)}")

    if "question_type" not in frame.columns:
        frame["question_type"] = "Unknown"

    dataset_frame = frame[["prompt", "answer", "question_type"]].copy()
    dataset_frame["prompt"] = dataset_frame["prompt"].map(build_prompt)
    dataset_frame = dataset_frame.rename(columns={"answer": "ground_truth"})
    return Dataset.from_pandas(dataset_frame, preserve_index=False)


def load_trainable_model(config: ReasoningGRPOProjectConfig):
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    if config.sft_adapter_path:
        return PeftModel.from_pretrained(
            base_model,
            config.sft_adapter_path,
            is_trainable=True,
        )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(base_model, peft_config)


def has_boxed_answer(text: str) -> bool:
    return bool(re.search(r"\\boxed\{[^}]+\}", text))


def normalized_reasoning_prefix(text: str) -> str:
    return re.sub(r"\\boxed\{[^}]+\}", "", text).strip().lower()


def build_reward_functions(config: ReasoningGRPOProjectConfig):
    def answer_reward(completions, ground_truth, **kwargs):
        scores = []
        for completion, gold in zip(completions, ground_truth):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            scores.append(config.answer_reward_weight if verify(gold, text) else 0.0)
        return scores

    def format_reward(completions, **kwargs):
        scores = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            scores.append(config.format_reward_weight if has_boxed_answer(text) else 0.0)
        return scores

    def question_type_reward(completions, question_type, **kwargs):
        scores = []
        for completion, expected_type in zip(completions, question_type):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            hit = expected_type.lower() in text.lower()
            scores.append(config.question_type_reward_weight if hit else 0.0)
        return scores

    def consistency_reward(completions, ground_truth, **kwargs):
        scores = []
        for completion, gold in zip(completions, ground_truth):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            extracted = extract_boxed_answer(text)
            non_empty_reasoning = bool(normalized_reasoning_prefix(text))
            consistent = non_empty_reasoning and verify(gold, extracted)
            scores.append(config.consistency_reward_weight if consistent else 0.0)
        return scores

    def length_penalty(completions, **kwargs):
        scores = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            scores.append(-config.excessive_length_penalty if len(text) > 1200 else 0.0)
        return scores

    return [
        answer_reward,
        format_reward,
        question_type_reward,
        consistency_reward,
        length_penalty,
    ]


def train(config: ReasoningGRPOProjectConfig) -> None:
    output_dir = ensure_dir(Path(config.output_dir))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_project_dataset(Path(config.train_file))
    model = load_trainable_model(config)

    trainer_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        logging_steps=1,
        save_strategy="steps",
        save_steps=20,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=build_reward_functions(config),
        args=trainer_config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Future-stage GRPO scaffold. This is a logical next step and is not expected to run locally without sufficient model infrastructure."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.train_file is not None:
        config.train_file = args.train_file
    if args.base_model_path is not None:
        config.base_model_path = args.base_model_path
    if args.sft_adapter_path is not None:
        config.sft_adapter_path = args.sft_adapter_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    train(config)


if __name__ == "__main__":
    main()
