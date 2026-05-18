# Results

这个目录用于放实验结果、误差分析和阶段性结论。

当前还没有真实训练结果被写入仓库，因此这里先提供结构模板，方便你后续补齐。

## 建议至少记录的内容

1. 数据版本
2. 训练配置
3. train/test 切分规则
4. overall accuracy
5. 按题型划分的 accuracy
6. 典型错误样本
7. 你对下一步优化方向的判断

## 当前文件

- [`experiment_log_template.md`](experiment_log_template.md): 实验记录模板

## 建议展示方式

可以按下面顺序组织：

1. Baseline
2. Answer-only SFT
3. Reasoning + answer SFT
4. Refined reasoning SFT
5. Future GRPO plan

