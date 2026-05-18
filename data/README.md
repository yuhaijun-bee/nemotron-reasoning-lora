# Data Layout

本目录只保留项目约定，不默认公开实际比赛数据。

## 目录说明

- `raw/`: 原始训练集与测试集
- `processed/`: reasoning 标注、审计、修订、严格切分等中间结果
- `samples/`: 小样本调试数据

## 建议本地放置的文件

```text
data/
├─ raw/
│  ├─ train.csv
│  └─ test.csv
├─ processed/
│  ├─ my_train_sample_500_boosted.csv
│  ├─ train_reasoning_rigorous.csv
│  ├─ train_reasoning_rigorous_audit.csv
│  ├─ train_reasoning_refined.csv
│  └─ ...
└─ samples/
   └─ train_sample_500.csv
```

## 为什么默认不提交 CSV

当前仓库面向 GitHub 公开发布，因此默认通过 `.gitignore` 忽略实际数据文件，避免把比赛数据、衍生数据或本地实验数据直接上传到公共仓库。

