# 01 · CIFAR-10 图像分类

> 状态：**🚧 骨架已就绪，代码待写**
> 对应笔记：[图像分类](../../notes/02-cnn/image-classification.md)、[ConvNet 架构范式](../../notes/02-cnn/convnet-patterns.md)

## 为什么选这个项目

| 理由 | 说明 |
|:--|:--|
| 完整闭环 | 数据增强 → CNN / 迁移学习 → 训练 → 评估 → 混淆矩阵，一条链走完 |
| 成本低 | 170MB 自动下载，CPU 也能 10 分钟出结果，RTX 4060 上 5–8 分钟跑完 |
| 对比性强 | 小型 CNN（从零训）与 ResNet18（迁移学习）同题对比，差距一目了然 |
| 概念覆盖广 | 数据增强、归一化、BatchNorm、ResNet、学习率调度、评估指标全覆盖 |

## 数据集

- **CIFAR-10**：10 类（飞机/汽车/鸟/猫/鹿/狗/蛙/船/卡车/货车），60,000 张 32×32 彩色图，
  训练 50,000 / 测试 10,000
- 来源：`torchvision.datasets.CIFAR10`，首次运行自动下载到 `data/`
- 归一化：`mean=(0.4914, 0.4822, 0.4465)`、`std=(0.2470, 0.2435, 0.2616)`（各通道统计量）

## 环境

```bash
# CPU 冒烟测试
pip install -r ../../requirements.txt

# GPU（RTX 4060 Laptop）
pip install -r ../../requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu124
```

自检：`python ../../scripts/check_env.py`

## 计划中的代码结构

```
src/
├── dataset.py     # 下载、归一化、训练增强（随机裁剪+水平翻转）、DataLoader
├── model.py       # build_cnn() / build_resnet18()
├── train.py       # 种子固定、训练循环、每轮验证、save best、TensorBoard
├── evaluate.py    # 混淆矩阵、分类报告、错误样本导出
├── predict.py     # 单图 Top-3 概率
└── utils.py       # set_seed / timer / device 检测 / 指标收集
```

## 计划中的运行方式

```bash
# 1) CPU 冒烟：确认全流程无 bug
python src/train.py --config config.yaml --model cnn --epochs 1 --device cpu

# 2) GPU 完整训练
python src/train.py --config config.yaml --model cnn     --epochs 30 --device cuda --amp
python src/train.py --config config.yaml --model resnet18 --epochs 20 --device cuda --amp

# 3) 评估与可视化
python src/evaluate.py --ckpt outputs/best_cnn.pt

# 4) 单图推理
python src/predict.py --ckpt outputs/best_cnn.pt --image samples/0064.png
```

## 预期结果（待实测填入）

| 指标 | 小型 CNN（30 epoch） | ResNet18 迁移学习（20 epoch） |
|:--|:--|:--|
| 测试集准确率 | 68% – 75% | 90% – 93% |
| 单 epoch 用时（RTX 4060） | 8 – 15 s | 12 – 20 s |
| 总训练时长 | 5 – 8 min | 5 – 7 min |

> 实测完成后用真实数字替换上表，并在下方「实验结论」记录分析。

## 实验结论

*（待填：哪个模型好在哪、过拟合表现、错误集中在哪些类别、改进尝试与效果）*

## 产出物

- `outputs/metrics.json` —— 指标汇总
- `outputs/training_curves.png` —— 训练/验证损失与准确率曲线
- `outputs/confusion_matrix.png` —— 混淆矩阵
- `outputs/best_*.pt` —— 最优权重（**已 gitignore，不入库**）
