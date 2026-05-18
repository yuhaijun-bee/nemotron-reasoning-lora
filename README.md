# NVIDIA Nemotron Reasoning LoRA Project

## Kaggle Result

The following screenshots show a recorded competition submission result for this project.

### Leaderboard Snapshot
![Leaderboard Snapshot](https://github.com/user-attachments/assets/e78cc8fd-a9b0-46c2-98a8-32aac43586ee)

### Submission Score
![Submission Score](https://github.com/user-attachments/assets/02a0c0e7-90e4-4472-b232-66f0453bb903)

一个围绕 `NVIDIA Nemotron` 的推理微调项目整理版，主线覆盖：

1. Wonderland 题目的题型识别与 reasoning 标注
2. 从初版标注到 rigorous 标注再到 refined 标注的逐步收敛
3. 面向 SFT/LoRA 的训练数据构建
4. 以 `reasoning_lora.py` 为最终训练入口的 Nemotron LoRA 微调
5. 基于现有 SFT 基线预留后续 `GRPO` 强化学习代码骨架

这个仓库的重点不是假装“一键完整复现全部大模型训练”，而是把这次实践整理成一个结构完整、叙事清楚、方便继续迭代和展示的项目。

## Project Overview

这个项目的核心问题是：  
如何把 Wonderland 风格的规则推理题，从原始问答数据，整理成可用于 reasoning 微调的数据流，再进一步连接到 LoRA 与后续 RL 优化。

当前仓库已经形成一条明确主线：

- 数据理解：识别题型，拆分不同规则任务
- 数据构造：为每条样本生成可监督的 reasoning
- 数据修订：对欠确定、模板化、跳步明显的 reasoning 做进一步处理
- 监督训练：用 `reasoning_lora.py` 作为最终 SFT/LoRA 训练入口
- 后续扩展：保留 `GRPO` 训练骨架，用于下一阶段 reward-based optimization

## What Is Final

如果只看一个入口，终版就是：

- 最终训练入口：[`reasoning_lora.py`](reasoning_lora.py)

对应实现文件：

- SFT/LoRA：[`src/reasoning_nemotron/training/reasoning_lora.py`](src/reasoning_nemotron/training/reasoning_lora.py)
- GRPO 骨架：[`src/reasoning_nemotron/training/grpo_reasoning.py`](src/reasoning_nemotron/training/grpo_reasoning.py)

## Repository Layout

```text
.
├─ configs/                         # 训练配置示例
├─ data/
│  ├─ raw/                          # 原始数据（本地保留，默认不提交）
│  ├─ processed/                    # 标注/切分后的数据（本地保留，默认不提交）
│  ├─ samples/                      # 小样本调试数据（本地保留，默认不提交）
│  └─ README.md
├─ docs/
│  ├─ project_roadmap.md            # 项目阶段说明
│  └─ notes/                        # 原始实验笔记
├─ examples/                        # 推理与调用示例
├─ results/                         # 实验记录模板与结果展示区
├─ src/
│  └─ reasoning_nemotron/
│     ├─ data/                      # 数据处理与 reasoning 标注
│     ├─ evaluation/                # 本地评估与审计
│     ├─ inference/                 # LoRA 推理脚本
│     ├─ training/                  # SFT/LoRA 与 GRPO
│     └─ legacy/                    # 早期探索脚本存档
├─ reasoning_lora.py                # 公开展示时的主入口
├─ requirements.txt
└─ pyproject.toml
```

## Pipeline

这套项目现在的逻辑可以概括为：

```text
raw train.csv
   -> annotate_reasoning
   -> annotate_reasoning_rigorous
   -> refine_underdetermined_reasoning
   -> split_single_item_train_test
   -> reasoning_lora.py (SFT/LoRA)
   -> grpo_reasoning.py (future stage)
```

## Environment Setup

建议优先使用可编辑安装：

```bash
pip install -e .
```

或者直接安装依赖：

```bash
pip install -r requirements.txt
```

## Data Policy

为了方便公开发布到 GitHub，当前 `.gitignore` 默认忽略：

- `data/raw/*.csv`
- `data/processed/*.csv`
- `data/samples/*.csv`

也就是说：

- 仓库保留数据目录结构
- 本地保留真实 CSV
- 公共仓库不直接上传比赛原始数据和衍生数据

具体约定见 [`data/README.md`](data/README.md)。

## Recommended Workflow

### 1. 生成初版 reasoning 标注

```bash
python -m reasoning_nemotron.data.annotate_reasoning
```

### 2. 生成更严格的 reasoning 标注

```bash
python -m reasoning_nemotron.data.annotate_reasoning_rigorous
```

### 3. 对欠确定样本做进一步修订

```bash
python -m reasoning_nemotron.data.refine_underdetermined_reasoning
```

### 4. 构造严格 train/test 切分

```bash
python -m reasoning_nemotron.data.split_single_item_train_test
```

### 5. 运行最终 SFT/LoRA 训练

```bash
python reasoning_lora.py --config configs/reasoning_lora.json
```

### 6. 后续阶段接入 GRPO

```bash
python -m reasoning_nemotron.training.grpo_reasoning --config configs/grpo_reasoning.json
```

## Main Components

### Data Annotation

- [`src/reasoning_nemotron/data/annotate_reasoning.py`](src/reasoning_nemotron/data/annotate_reasoning.py)  
  初版题型识别与 reasoning 标注

- [`src/reasoning_nemotron/data/annotate_reasoning_rigorous.py`](src/reasoning_nemotron/data/annotate_reasoning_rigorous.py)  
  更严格的推理构造与 underdetermined 标记

- [`src/reasoning_nemotron/data/refine_underdetermined_reasoning.py`](src/reasoning_nemotron/data/refine_underdetermined_reasoning.py)  
  结合外部参考规则，对欠确定样本进一步修订

### Evaluation

- [`src/reasoning_nemotron/evaluation/audit_reasoning.py`](src/reasoning_nemotron/evaluation/audit_reasoning.py)  
  审计 reasoning 中的跳步、答案不一致、格式问题

- [`src/reasoning_nemotron/evaluation/local_metric.py`](src/reasoning_nemotron/evaluation/local_metric.py)  
  本地 answer matching 与 prediction scoring

### Training

- [`src/reasoning_nemotron/training/reasoning_lora.py`](src/reasoning_nemotron/training/reasoning_lora.py)  
  终版 SFT/LoRA 训练实现

- [`src/reasoning_nemotron/training/grpo_reasoning.py`](src/reasoning_nemotron/training/grpo_reasoning.py)  
  下一阶段 GRPO 代码骨架

## Why `reasoning_lora.py` Matters

`reasoning_lora.py` 是这次实践里的最终训练入口。它现在已经从原来的 Kaggle 实验脚本，整理成了：

- 可读的配置结构
- 相对路径输入输出
- 统一的数据格式化逻辑
- 实验配置落盘
- LoRA adapter 自动打包为 zip

这意味着它既能继续作为你自己的训练主线，也能作为公开仓库的主展示入口。

## GRPO Status

这个项目当前还没有真正进入强化学习训练阶段，但已经补了完整的下一阶段代码骨架：

- 文件：[`src/reasoning_nemotron/training/grpo_reasoning.py`](src/reasoning_nemotron/training/grpo_reasoning.py)
- 定位：SFT 之后的 reward-based optimization 起点
- 状态：结构完整，但默认不承诺本地直接跑通

目前设计的 reward 主要包括：

- answer correctness
- boxed answer format
- question type consistency
- reasoning and answer consistency
- excessive length penalty

## Results And Examples

我已经额外补了两个用于发布展示的区域：

- 结果区：[`results/README.md`](results/README.md)
- 实验记录模板：[`results/experiment_log_template.md`](results/experiment_log_template.md)
- 示例区：[`examples/README.md`](examples/README.md)
- 推理示例模板：[`examples/inference_example.md`](examples/inference_example.md)

你后续只需要把真实实验结论、截图、误差分析填进去，仓库观感会完整很多。

## Current Limitations

- 没有在本地完整部署 Nemotron 大模型
- GRPO 仍是代码骨架，不是已完成实验
- 当前公开仓库默认不包含真实训练数据
- 结果区目前是展示模板，等你填入真实实验记录后会更完整

## Next Practical Improvements

1. 在 `results/` 中补齐你自己的实验配置和准确率结果
2. 用 `examples/` 放几条真实推理输入输出样例
3. 把 `docs/notes/` 进一步整理成正式实验日志
4. 在有合适环境后验证并迭代 `GRPO` 模块

## One-Line Summary

这是一个从 reasoning 标注、严格化数据构建、Nemotron LoRA 微调，到后续 GRPO 过渡接口都已整理清楚的项目骨架。
