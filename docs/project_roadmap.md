# Project Roadmap

## Stage 1: 当前已经完成的主线

1. 题型识别
2. reasoning 标注
3. rigorous reasoning 修订
4. underdetermined 样本进一步修订
5. 单样本严格切分
6. SFT/LoRA 最终训练入口整理

## Stage 2: 下一步最合理的推进

1. 固定一套严格评估集
2. 用当前 refined 数据先稳定 SFT 基线
3. 把模型输出格式统一为“reasoning + final boxed answer”
4. 接入 rule-based reward
5. 再进入 GRPO

## 为什么先 SFT，再 GRPO

如果 SFT 阶段的数据格式、标签质量、答案抽取和本地评估都还不稳定，直接做 RL 很容易把问题放大。

更合理的顺序是：

- 先把监督数据管线整理干净
- 先得到一个稳定可复现的 LoRA 基线
- 再让 GRPO 去优化答案正确率、格式稳定性和 reasoning 一致性

## 当前 GRPO 文件的定位

`src/reasoning_nemotron/training/grpo_reasoning.py`

它是“下一阶段代码骨架”，不是已经验证完成的生产训练脚本。

也就是说：

- 结构和奖励逻辑已经给出
- 可以直接作为后续实验起点
- 但真正跑通仍然依赖大模型部署、显存、采样吞吐和 RL 调参环境

