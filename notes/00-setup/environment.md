---
chapter: 3
book_section: "Ch3 · TensorFlow、PyTorch、JAX 与 Keras 入门"
tags: [环境, CUDA, 踩坑]
env: [python 3.12.15, pytorch 2.13.0+cu126, rtx4060-laptop]
updated: 2026-10-21
status: draft
---

# 深度学习环境搭建

> 记录我这台机器的深度学习环境：是什么、怎么验证、以及装的时候踩过的坑。

## 0. TL;DR

- 我的环境：conda 环境名 `dl`（装在 D 盘），Python 3.12.15，PyTorch 2.13.0+cu126。
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU（sm_8.9，8.0 GB 显存），驱动 571.96，CUDA 构建版本 12.6，`torch.cuda.is_available()` 返回 `True`。
- 环境验证入口：`python scripts/check_env.py`，一次打印 Python / 各依赖版本 / CUDA 可用性 / 设备名 / 显存占用。
- 三层关系一句话：**驱动让 Windows 认识显卡 → CUDA 让代码能指挥显卡 → PyTorch 帮你定义并训练网络。** 环境出问题就从下往上查。

## 1. 白话解释

装深度学习环境，本质是让电脑具备「训练神经网络」的能力。这件事需要三样东西配合，缺一不可。

### 1.1 PyTorch —— 真正干活的

这是我直接打交道的框架。我写 `loss.backward()`、`model.add(...)` 这些代码，PyTorch 负责把网络跑起来、算出损失、算梯度、然后更新参数。

它只管算法，不管硬件。PyTorch 本身并不知道我这台机器有没有显卡。

### 1.2 CUDA —— 显卡和代码之间的翻译层

显卡（GPU）擅长的是「同时做几百万次简单乘法」，而这恰好符合神经网络的特征：绝大部分计算量都在矩阵乘法上。

但显卡听不懂我写的 PyTorch 代码。CUDA 就是那层翻译，它把 PyTorch 的指令翻译成显卡能执行的东西，再把结果收回来。

它负责调度，不负责思考。

### 1.3 NVIDIA 驱动 —— 让 Windows 认识显卡的那层胶水

买了新显卡插进电脑，Windows 未必能直接用它——显卡的指令需要有人翻译成 Windows 能调度的东西，驱动干的就是这个活。

所以排错要从下往上：**PyTorch 报错了，先看 CUDA 配得对不对；CUDA 用不了，先看驱动装没装。** 这是环境问题的固定检查顺序。

### 1.4 一句话串起来

> 驱动让 Windows 认识显卡 → CUDA 让我写的代码能指挥显卡 → PyTorch 帮我定义和训练网络。

### 1.5 我装环境时遇到的两个问题

**问题一：以为 `pip install torch` 就够了。**

装完发现显卡完全没被用上——代码能跑，但慢得离谱。原因是我装的是 CPU 版。PyTorch 官方对同一套代码发布了很多版本（CPU 版、各个 CUDA 版本的 GPU 版），下载页面看着差不多，但装完的能力完全不同。所以这一步必须显式指定 CUDA 版本：

```bash
# 显式指定 CUDA 版本安装，而不是用默认源
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**问题二：一个环境里只能有一套 torch。**

如果 conda 和 pip 都装过 torch，`import torch` 到底加载哪一个是不确定的，表现为版本号莫名其妙、或者改动不生效。检查方法：

```bash
pip list | grep -i torch     # 出现多行说明混装了
conda list | grep -i torch   # 同样查一遍
```

一个环境里只能有一套，混用了就彻底卸干净重装。

## 2. 我的环境清单

| 项 | 版本 | 备注 |
|:--|:--|:--|
| OS | Windows 11 (22621) | |
| Python | 3.12.15 | conda 环境 `dl`，路径 `D:\conda\envs\dl\` |
| PyTorch | 2.13.0+cu126 | 注意是 cu126 不是 cu124 |
| torchvision | 0.28.0+cu126 | 与 torch 版本配套 |
| numpy | 2.5.4 | 大版本跳跃，见坑 3 |
| scipy | 1.18.1 | |
| scikit-learn | 1.9.1 | |
| pandas | 3.0.6 | 大版本跳跃，见坑 3 |
| matplotlib | 3.11.2 | |
| PyYAML / tqdm | 6.0.3 / 4.70.1 | |
| seaborn | 未安装 | 可选，画图增强 |
| tensorboard | 未安装 | 可选，训练曲线可视化 |
| 驱动 | 571.96 | `nvidia-smi` 显示 |
| 驱动支持的 CUDA 上限 | 12.8 | `nvidia-smi` 显示 |
| PyTorch 编译时 CUDA | 12.6 | `torch.version.cuda` |
| GPU | RTX 4060 Laptop（sm_8.9） | 8.0 GB 显存 |

## 3. 三层版本号的关系

`nvidia-smi` 里的 `CUDA Version: 12.8` 和 `torch.version.cuda` 的 `12.6` 是**两个不同的东西**，最容易混：

- `nvidia-smi` 显示的是**驱动最高能支持到哪个 CUDA 版本**（能力上限）
- `torch.version.cuda` 是**我这个 torch 包编译时用哪个 CUDA 版本**（实际需求）

方向只要满足「驱动 ≥ 需求」就能跑：驱动 571.96 支持到 12.8，torch 需要 12.6，够用。**反了就会报错**（见坑 1）。

类比：高版本播放器能播低版本视频，反过来不行。

## 4. 怎么验证环境

```bash
python scripts/check_env.py
```

预期最后一行输出「环境就绪，可以开始跑示例。」，退出码 0。脚本里的未安装项会显式标「未安装」，不要跳过——那说明它如实反映了现状。

最小验证代码：

```python
import torch
print(torch.cuda.is_available())   # True 才算 GPU 可用
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())   # (空闲字节, 总字节)
```

## 5. 关键 API 速查

| API | 作用 |
|:--|:--|
| `torch.cuda.is_available()` | GPU 是否可用 |
| `torch.cuda.device_count()` | 有几张 GPU |
| `torch.cuda.get_device_name(0)` | 第 0 张卡的名字 |
| `torch.cuda.get_device_capability(0)` | 计算能力（如 sm_89） |
| `torch.cuda.mem_get_info()` | (空闲, 总) 显存字节数 |
| `torch.version.cuda` | 本机 torch 编译时的 CUDA 版本 |
| `torch.cuda.OutOfMemoryError` | 显存不足的异常类型 |

## 6. 常见坑与排查

### 坑 1：`no kernel image is available for execution on the device`

- **症状**：运行时报这个错，GPU 完全没参与计算，耗时明显变长
- **原因**：NVIDIA 驱动版本**低于** PyTorch 编译时所需的 CUDA 版本。驱动和 CUDA 是两个独立版本，前者是后者的能力上限
- **解法**：`nvidia-smi` 看 Driver Version，要求 ≥ `torch.version.cuda` 对应的版本号；不够就升级显卡驱动，或改装低 CUDA 版本的 torch
- **延伸**：这是环境问题不是代码问题。出现时先查版本，别急着改模型

### 坑 2：显存明明还剩很多，却报CUDA out of memory

- **症状**：`torch.cuda.OutOfMemoryError`，但 `nvidia-smi` 显示还剩大把显存
- **原因**：GPU 常驻占用约 894 MiB，但利用率 0%、功耗仅 12W——说明**有进程占着显存却没在计算**，通常是浏览器硬件加速或后台程序
- **解法**：先 `nvidia-smi` 看 Processes 表格找出占用进程（Windows 上若该表格为空，去任务管理器「性能 → GPU」看），关掉再重跑；实在不够再降 `batch_size`
- **延伸**：8 GB 显存的卡跑 CIFAR-10，`batch_size` 建议从 64 或 128 起步

### 坑 3：跟着书上的代码跑，报 `AttributeError`

- **症状**：书或网上的老代码跑起来报 `np.float_`、`np.NaN` 之类属性不存在，或 sklearn 参数名对不上
- **原因**：numpy 2.5 / pandas 3.0 / scikit-learn 1.9 都是**大版本跳跃后的新版本**，而老代码按numpy 1.x / pandas 1.x 的 API 写
- **解法**：优先**改代码适配新 API**（`np.float_` → `np.float64`），而不是降级环境。降级会带来更多兼容问题
- **延伸**：这是新环境的优势而非劣势。看到报错先怀疑版本差异，不要以为是自己的代码写错了

### 坑 4：PowerShell 里报 `head` 不是可识别的命令

- **症状**：`head : 无法将"head"项识别为 cmdlet`
- **原因**：`head` 是 Linux/macOS 的命令，PowerShell 没有。提示符是 `PS` 就说明当前是 PowerShell
- **解法**：用 `Select-Object -First 8` 替代 `head -8`；或切到 Git Bash
- **延伸**：建议 git 命令统一在 Git Bash 里跑（原生支持），代码运行用哪个 shell 都行

## 7. 动手练习

1. 用 `check_env.py` 记下当前空闲显存，然后在浏览器多开几个标签页再跑一次，观察数值变化。

   <details>
   <summary>参考答案</summary>

   空闲显存会下降，说明浏览器在用 GPU 做硬件加速（视频解码、页面渲染）。这就解释了坑 2 里「利用率 0% 但显存被占」的现象。

   </details>

2. 故意在一个没装 torch 的环境里 `import torch`，记录报错信息，对照坑 3 思考两者有什么本质不同。

   <details>
   <summary>参考答案</summary>

   缺包报 `ModuleNotFoundError`（模块根本不存在），版本不匹配报 `AttributeError`（模块存在但没这个属性）或参数不匹配的 `TypeError`。前者是「没装」，后者是「装了但版本对不上」，排查方向不同。

   </details>

## 8. 参考资料

- 《Deep Learning with Python》第 2 版，Ch3
- [PyTorch 官方安装指南（含各 CUDA 版本对应表）](https://pytorch.org/get-started/locally/)
- [NVIDIA 驱动与 CUDA 版本对应关系](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [PyTorch 显存管理 FAQ](https://pytorch.org/docs/stable/notes/cuda.html)

## 9. 自检清单

- [ ] 能说出驱动、CUDA、PyTorch 三者各自负责什么
- [ ] 能解释 `nvidia-smi` 的 CUDA Version 和 `torch.version.cuda` 有什么区别
- [ ] 能独立跑 `check_env.py` 并读懂全部输出
- [ ] 能说出至少两个常见坑的症状与解法
- [ ] 知道自己的环境里有没有混装两套 torch