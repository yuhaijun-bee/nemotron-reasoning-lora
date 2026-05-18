# tokenizer
def tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# dataset
datasets = [
    [
        {"role": "system", "content": "You are a logical reasoning expert. You think step-by-step and prioritize correctness and structured reasoning."},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"}
    ],
    [
        {"role": "system", "content": "You are a logical reasoning expert. You think step-by-step and prioritize correctness and structured reasoning."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}
    ]
]

# tokenize
def tokenize(datasets, tokenizer):
    encodings = tokenizer(
        datasets,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    return encodings, input_ids, attention_mask

# labels
def get_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100 # 将padding部分的标签设置为-100，这样在计算损失时会被忽略
    #not_answer_mask 之后会将所有非答案部分的标签设置为-100，这样在计算损失时也会被忽略，只关注答案部分的预测
    #这就意味着我们缺少了过程推理的监督信号，模型在训练过程中可能无法学习到如何进行推理和生成答案的过程。
    return labels

# dataloader
from torch.utils.data import DataLoader, TensorDataset
def get_dataloader(input_ids, attention_mask, labels):
    dataset = TensorDataset(
        input_ids,
        attention_mask,
        labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle = True
    )
    return dataloader

# model
def model(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model

# optimizer
def optimizer(model):
    from torch.optim import AdamW
    optimizer = AdamW(
        model.parameters(), # ->后期只更新LoRA参数
        lr=5e-5
    )
    return optimizer

# training loop
def train(model, dataloader, optimizer, epochs=3):  
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")