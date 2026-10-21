# dl-notes · 深度学习笔记库

> 系统学深度学习的长期沉淀。以《Deep Learning with Python》（第 2 版，Keras 为主线）为线索，
> 把 20 章内容重组成 **概念直觉 → 数学形式 → 最小可复现代码 → 真实踩坑** 四层可复用的笔记。

## 这个仓库想解决什么

初试 408 之外，复试阶段「机器学习与深度学习原理」是被高频追问的部分；而 Transformer、
扩散模型这类前沿模型更新极快，只追论文容易抓不住主干，两头都不牢。

本库的定位是**把原理讲透、把代码跑通**：每一篇笔记都要能回答"为什么这样设计"，
也必须能复制粘贴就跑起来；每一个踩过的坑都写下来，附症状与解法，
避免半年后的自己重蹈覆辙。

- **主线 PyTorch，旁注 Keras** —— 原书用 Keras（易读），论文与工业用 PyTorch（务实），同一概念两栏对照
- **每篇固定九段式** —— TL;DR / 白话解释 / 数学形式 / 张量形状流 / 最小可运行代码 / API 速查 / 常见坑 / 动手练习 / 参考资料
- **坑位四段式** —— 症状 → 原因 → 解法 → 延伸，可按症状检索
- **持续更新** —— 进度见下方目录索引，每完成一个阶段打一个 tag

## 目录索引

| 阶段 | 主题 | 对应章节 | 笔记 | 状态 |
|:--|:--|:--|:--|:--|
| S0 | 环境与工具 | Ch3 | [environment](notes/00-setup/environment.md) | 🚧 进行中 |
| S0 | 框架对比 | Ch3 | [frameworks-overview](notes/00-setup/frameworks-overview.md) | ⬜ 待写 |
| S0 | GPU / CUDA 排错 | Ch3 | [gpu-cuda-troubleshooting](notes/00-setup/gpu-cuda-troubleshooting.md) | ⬜ 待写 |
| S1 | 深度学习是什么 | Ch1 | [what-is-deep-learning](notes/01-foundations/what-is-deep-learning.md) | ⬜ 待写 |
| S1 | 神经网络数学构件 | Ch2 | [math-building-blocks](notes/01-foundations/math-building-blocks.md) | ⬜ 待写 |
| S1 | 分类与回归 | Ch4 | [classification-and-regression](notes/01-foundations/classification-and-regression.md) | ⬜ 待写 |
| S1 | 机器学习基础 | Ch5 | [ml-fundamentals](notes/01-foundations/ml-fundamentals.md) | ⬜ 待写 |
| S1 | 通用工作流 | Ch6 | [universal-ml-workflow](notes/01-foundations/universal-ml-workflow.md) | ⬜ 待写 |
| S1 | Keras 深入 | Ch7 | [keras-deep-dive](notes/01-foundations/keras-deep-dive.md) | ⬜ 待写 |
| S2 | 图像分类 | Ch8 | [image-classification](notes/02-cnn/image-classification.md) | ⬜ 待写 |
| S2 | ConvNet 架构范式 | Ch9 | [convnet-patterns](notes/02-cnn/convnet-patterns.md) | ⬜ 待写 |
| S2 | 卷积网络可解释性 | Ch10 | [convnet-interpretation](notes/02-cnn/convnet-interpretation.md) | ⬜ 待写 |
| S2 | 图像分割 | Ch11 | [image-segmentation](notes/02-cnn/image-segmentation.md) | ⬜ 待写 |
| S2 | 目标检测 | Ch12 | [object-detection](notes/02-cnn/object-detection.md) | ⬜ 待写 |
| S3 | 时间序列预测 | Ch13 | [timeseries-forecasting](notes/03-sequence-and-text/timeseries-forecasting.md) | ⬜ 待写 |
| S3 | 文本分类 | Ch14 | [text-classification](notes/03-sequence-and-text/text-classification.md) | ⬜ 待写 |
| S3 | 语言模型与 Transformer | Ch15 | [language-models-transformer](notes/03-sequence-and-text/language-models-transformer.md) | ⬜ 待写 |
| S3 | 文本生成 | Ch16 | [text-generation](notes/03-sequence-and-text/text-generation.md) | ⬜ 待写 |
| S4 | 图像生成 | Ch17 | [image-generation](notes/04-generation/image-generation.md) | ⬜ 待写 |
| S4 | 真实场景最佳实践 | Ch18 | [best-practices-real-world](notes/05-practice/best-practices-real-world.md) | ⬜ 待写 |
| S4 | 读后感 | Ch19–20 | [reflections](notes/05-practice/reflections.md) | ⬜ 待写 |

图例：✅ 已完成 ｜ 🚧 进行中 ｜ ⬜ 待写 ｜ 📌 草稿

## 学习路线

| 阶段 | 主题 | 章节 | 为什么这个顺序 |
|:--|:--|:--|:--|
| **S0** | 环境与工具 | Ch3 | 先能跑起来，再谈原理 |
| **S1** | 基础与通用工作流 | Ch1/2/4/5/6/7 | Ch6 是全书提纲，**先读它**能省大量重复理解 |
| **S2** | 视觉：CNN | Ch8–12 | 本书最强的部分，对应主线示例项目 |
| **S3** | 序列与文本 | Ch13–16 | Ch13 时序预测接交通流方向，优先级上调 |
| **S4** | 生成与进阶 | Ch17–20 | 拓宽视野，**可以永远不写完**，不阻塞主线 |

建议节奏：每天 1–2 小时，一次读 1–2 小时书（控制在 30 页内），读完立刻写笔记，
趁热把坑记下来。

## 使用说明

### 本地预览

```bash
git clone https://github.com/<你的用户名>/dl-notes.git
cd dl-notes
```

用 VS Code 打开任意 `.md` 文件，`Ctrl+Shift+V` 预览（支持数学公式需装 Markdown+Math 扩展）；
或用 `mdcat notes/**/*.md` 在终端看。

### 新增一篇笔记

```bash
python scripts/new_note.py --stage 01-foundations --title "神经网络数学构件"
# 会在对应目录生成带完整九段式骨架的 markdown
```

### 环境自检

```bash
python scripts/check_env.py
# 打印 Python / PyTorch / CUDA / 显卡 / 各依赖版本
```

### 贡献

欢迎 fork + PR。新增笔记请遵守 `notes/_template.md` 的九段式结构，
代码块务必标注语言，参考链接优先用官方文档永久链接。

## 示例项目

| 项目 | 目录 | 内容 | 预期结果 |
|:--|:--|:--|:--|
| CIFAR-10 图像分类 | [examples/01-cifar10-cnn](examples/01-cifar10-cnn/) | 小型 CNN vs ResNet18 迁移学习全流程 | 小 CNN ≈ 68–75%，ResNet18 ≈ 90–93% |

更多示例见 [examples/README.md](examples/README.md)。

## 环境要求

- Python 3.10+
- CPU 基线：`pip install -r requirements.txt`
- GPU（RTX 4060 Laptop）：`pip install -r requirements-gpu.txt`

## License

[MIT](LICENSE) © 2026 李玉彤
