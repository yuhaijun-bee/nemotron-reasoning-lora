from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import types
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from reasoning_nemotron.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ensure_dir

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional runtime dependency
    kagglehub = None


SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put the final answer in \\boxed{}."
)


@dataclass
class ReasoningLoraConfig:
    model_handle: str = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    model_path: str | None = None
    train_file: str = str(PROCESSED_DATA_DIR / "my_train_sample_500_boosted.csv")
    output_dir: str | None = None
    submission_zip: str | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_len: int = 512
    num_epochs: float = 1.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    use_bf16: bool = True


def running_on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists()


def load_config(config_path: str | None) -> ReasoningLoraConfig:
    config = ReasoningLoraConfig()
    if not config_path:
        return config

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key, value in payload.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown config field: {key}")
        setattr(config, key, value)
    return config


def finalize_paths(config: ReasoningLoraConfig) -> ReasoningLoraConfig:
    output_dir = Path("/kaggle/working/adapter") if running_on_kaggle() else ARTIFACTS_DIR / "checkpoints" / "reasoning_lora"
    submission_zip = (
        Path("/kaggle/working/submission.zip")
        if running_on_kaggle()
        else ARTIFACTS_DIR / "submissions" / "reasoning_lora_adapter.zip"
    )
    if config.output_dir is None:
        config.output_dir = str(output_dir)
    if config.submission_zip is None:
        config.submission_zip = str(submission_zip)
    return config


def pure_rmsnorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    eps: float = 1e-5,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    upcast: bool = True,
) -> torch.Tensor:
    del group_size, norm_before_gate
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)


def apply_runtime_workarounds() -> None:
    for _, module in list(sys.modules.items()):
        if hasattr(module, "rmsnorm_fn"):
            module.rmsnorm_fn = pure_rmsnorm_fn

    for module_name in [
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.cute",
        "mamba_ssm.ops.cute.mamba3",
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
    ]:
        sys.modules[module_name] = types.ModuleType(module_name)
    sys.modules["mamba_ssm.modules.mamba3"].Mamba3 = None

    source_ptxas = Path(
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
    )
    if not source_ptxas.exists():
        return

    target_ptxas = Path("/tmp/ptxas-blackwell")
    shutil.copy2(source_ptxas, target_ptxas)
    os.chmod(
        target_ptxas,
        os.stat(target_ptxas).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
    )

    import triton.backends.nvidia as nv_backend
    import triton.backends.nvidia.compiler as nv_compiler

    source_bin = Path(nv_backend.__file__).resolve().parent / "bin"
    target_bin = Path("/tmp/triton_nvidia_bin")
    shutil.copytree(source_bin, target_bin, dirs_exist_ok=True)
    for file_path in target_bin.iterdir():
        if file_path.is_file():
            os.chmod(
                file_path,
                os.stat(file_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
            )

    nv_backend.__file__ = str(target_bin.parent / "__init__.py")
    nv_compiler.get_ptxas_version = lambda arch: "12.0"
    os.environ["TRITON_PTXAS_PATH"] = str(target_ptxas)
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(target_ptxas)


def resolve_model_path(config: ReasoningLoraConfig) -> str:
    if config.model_path:
        return str(Path(config.model_path))
    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed. Provide --model-path or install kagglehub first."
        )
    return kagglehub.model_download(config.model_handle)


def format_chat_example(example: dict[str, str], tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["reasoning"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_training_dataset(train_file: Path, tokenizer: AutoTokenizer) -> Dataset:
    train_frame = pl.read_csv(train_file)
    required_columns = {"prompt", "reasoning"}
    missing = required_columns.difference(train_frame.columns)
    if missing:
        raise ValueError(f"Training file is missing required columns: {sorted(missing)}")

    dataset = Dataset.from_pandas(
        train_frame.select(["prompt", "reasoning"]).to_pandas(),
        preserve_index=False,
    )
    dataset = dataset.map(
        lambda row: format_chat_example(row, tokenizer),
        remove_columns=dataset.column_names,
    )
    return dataset


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_path: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    for module_name, module in sys.modules.items():
        if "modeling_nemotron_h" in module_name:
            module.is_fast_path_available = False
    return model


def attach_lora(model: AutoModelForCausalLM, config: ReasoningLoraConfig):
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    wrapped_model = get_peft_model(model, lora_config)
    wrapped_model.print_trainable_parameters()
    return wrapped_model


def save_run_config(config: ReasoningLoraConfig, output_dir: Path) -> None:
    ensure_dir(output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def package_adapter(output_dir: Path, submission_zip: Path) -> None:
    ensure_dir(submission_zip.parent)
    if submission_zip.exists():
        submission_zip.unlink()

    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in required_files:
            source_path = output_dir / filename
            if not source_path.exists():
                raise FileNotFoundError(f"Missing adapter artifact: {source_path}")
            archive.write(source_path, filename)


def train(config: ReasoningLoraConfig) -> None:
    config = finalize_paths(config)
    apply_runtime_workarounds()

    model_path = resolve_model_path(config)
    tokenizer = load_tokenizer(model_path)
    dataset = load_training_dataset(Path(config.train_file), tokenizer)
    model = attach_lora(load_model(model_path), config)

    output_dir = ensure_dir(Path(config.output_dir))
    submission_zip = Path(config.submission_zip)
    save_run_config(config, output_dir)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        bf16=config.use_bf16,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=config.max_seq_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    package_adapter(output_dir, submission_zip)

    print(f"train_file={config.train_file}")
    print(f"output_dir={output_dir}")
    print(f"submission_zip={submission_zip}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final SFT/LoRA training entrypoint for the reasoning project.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--submission-zip", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.train_file is not None:
        config.train_file = args.train_file
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.submission_zip is not None:
        config.submission_zip = args.submission_zip

    train(config)


if __name__ == "__main__":
    main()
