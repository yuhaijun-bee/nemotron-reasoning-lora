#进行PEFT微调，使用LoRA方法
#先实现chat_LoRA
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

model.train()
#数据格式标准化
from datasets import Dataset
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant."
    "Solve problems step-by-step."
    "Return only one final answer in \\boxed{}."
)
formatted_data = []
for _,row in final_data.iterrows():
    ans = str(row['answer']).strip()
    if not ans.startswith(r"\boxed{"):
        ans = f"\boxed{{{ans}}}"
    formatted_data.append({
        "messages":[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": row["prompt"]
            },
            {
                "role": "assistant",
                "content": ans
            }
        ]
    })
work_dataset = Dataset.from_list(formatted_data)

# LoRA配置
lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)   

#LoRA注入
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 打印可训练参数的数量和占比


# tokenize
#会将所有非答案部分的标签设置为-100，这样在计算损失时也会被忽略，只关注答案部分的预测
#这就意味着我们缺少了过程推理的监督信号，模型在训练过程中可能无法学习到如何进行推理和生成答案的过程。
def preprocess_function(data):
    messages = data["messages"]
    # 完整对话
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    # 不包含assistant answer
    context_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True
    )
    tokenized_full = tokenizer(
        full_text,
        truncation = True,
        padding = False,
        add_special_tokens = False
    )
    tokenized_context = tokenizer(
        context_text,
        truncation = True,
        padding = False,
        add_special_tokens = False
    )
    input_ids = tokenized_full['input_ids']
    labels = input_ids.copy()
    context_len = len(tokenized_context['input_ids'])
    labels[:context_len] = [-100] * context_len

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels
    }

preprocessed_dataset = work_dataset.map(preprocess_function)
final_dataset = preprocessed_dataset.remove_columns(["messages"])
# dataloader
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
final_dataset.set_format("torch")
collator = DataCollatorForSeq2Seq(
    tokenizer = tokenizer,
    model = model,
    padding = True
)
dataloader = DataLoader(
    final_dataset,
    batch_size = 2,
    shuffle = True,
    collate_fn = collator
)


# optimizer
from torch.optim import AdamW
optimizer = AdamW(
    model.parameters(), # ->后期只更新LoRA参数
    lr=5e-5
)


# training loop
for batch in dataloader:
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())