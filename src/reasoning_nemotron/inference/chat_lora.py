from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_nemotron.evaluation.local_metric import extract_boxed_answer


SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step and put the final answer in \\boxed{}."
)


def load_chat_model(base_model_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def build_chat_prompt(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def run_generation(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = build_chat_prompt(prompt)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][encoded.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompt against the trained LoRA adapter.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer, model = load_chat_model(args.base_model_path, args.adapter_path)
    response = run_generation(
        tokenizer=tokenizer,
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("response:")
    print(response)
    print()
    print(f"boxed_answer={extract_boxed_answer(response)}")


if __name__ == "__main__":
    main()
