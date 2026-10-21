# 示例项目索引

这里的每个项目都是**能直接跑起来**的完整代码，不是片段。目标是让你在读笔记的同时，
手上有东西能验证概念。

| 项目 | 难度 | 对应笔记 | 框架 | 一句话 |
|:--|:--|:--|:--|:--|
| [01-cifar10-cnn](01-cifar10-cnn/) | ★★ | [图像分类](../notes/02-cnn/image-classification.md)、[ConvNet 架构范式](../notes/02-cnn/convnet-patterns.md) | PyTorch | 从零训小型 CNN vs ResNet18 迁移学习，完整走通「数据增强 → 训练 → 评估 → 混淆矩阵」 |

## 通用约定

所有项目遵循同一套结构，方便形成肌肉记忆：

```
<project>/
├── README.md          # 环境 / 数据 / 运行方式 / 预期结果 / 结论
├── config.yaml        # 唯一超参来源，所有实验参数集中在这里
├── requirements.txt   # 依赖（GPU 版见 requirements-gpu.txt）
└── src/
    ├── dataset.py     # 数据管线：下载、预处理、增强、DataLoader
    ├── model.py       # 模型定义
    ├── train.py       # 训练主循环
    ├── evaluate.py    # 评估与可视化
    ├── predict.py     # 单样本 / 批量推理
    └── utils.py       # 种子、计时、GPU 检测、指标收集
```

## 原则

- **随机种子固定**：`utils.set_seed()` 统一设置 random / numpy / torch / PYTHONHASHSEED，
  保证同一配置能复现结果
- **权重不入库**：训练产出的 `*.pt` 已在 `.gitignore` 中排除；重要结果请在 README 里
  记录指标表格与曲线图
- **图表可复现**：绘图代码在 `src/` 内，输出到 `outputs/`，图片（PNG）可入库
- **先 CPU 冒烟**：先用 `--device cpu --epochs 1` 跑通全流程，再上 GPU 跑完整实验
