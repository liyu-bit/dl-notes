---
chapter: 3
book_section: "Ch3 · TensorFlow、PyTorch、JAX 与 Keras 入门"
tags: [框架对比, Keras, PyTorch, 广播]
env: [python 3.12.15, pytorch 2.13.0+cu126, tf未安装]
updated: 2026-10-21
status: draft
---

# 框架选型：TF / PyTorch / JAX / Keras 的关系

> 四个名字摆在一起，但它们不在同一层级。这篇理清各自的定位，以及本库为什么以 PyTorch 为主线。

## 0. TL;DR

- 这四个名字**不是并列的竞争者**。TensorFlow / PyTorch / JAX 是三个**计算引擎**（真正干活的后端），Keras 是架在它们之上的**接口层**。
- 本书《Deep Learning with Python》用 Keras 写作，但我的库以 **PyTorch 为主线、Keras 为对照**——论文与工业界用 PyTorch 更普遍，复试答原理时两边能互相印证。
- 我的机器**没装 TensorFlow**（实测报 `ModuleNotFoundError`），所以框架对比只能一手实测 PyTorch，其余按官方文档转述。
- 所有框架共用的底层概念是**张量广播**：`(3,1) + (1,4) → (3,4)`，实测通过。这才是「深度学习为什么能统一」的根本原因。

## 1. 白话解释

### 1.1 先把四个名字分个级

日常说法里它们被当成四个平级的「框架」放在一起比较，这是最容易踩的认知坑。实际上分两层：

| 层级 | 成员 | 干什么 |
|:--|:--|:--|
| **计算引擎（后端）** | TensorFlow、PyTorch、JAX | 真正执行矩阵运算、张量计算、GPU 调度 |
| **接口层** | Keras | 只提供易用的 API，把代码翻译给某个后端 |

一个类比：**引擎**像发动机，**Keras**像变速箱。你换引擎（换后端），车照样能开，只是要重新学变速箱的脾气。

### 1.2 三个引擎各自什么来路

- **TensorFlow**（Google, 2015）—— 产业落地最广，Android、推荐系统、各类部署场景大量在用。生态最庞大，代价是 API 相对啰嗦。
- **PyTorch**（Meta）—— 学术界与研究的事实标准。它的写法贴近 Python 本身，调试时可以打断点看中间变量，论文里90% 的深度学习代码是它。
- **JAX**（Google）—— 面向加速与可组合函数变换的研究型框架，配合 `jit` / `grad` / `vmap` 这类函数变换工具，擅长把整个计算过程编译优化。生态最年轻。

### 1.3 Keras 现在是独立的接口层

这是本篇最需要更新认知的一点：**Keras 已从 TensorFlow 内部剥离出来成为独立包**（Keras 3），它自己不做计算，只负责把你的代码翻译给后端。

据 Keras 3 官方文档，它的后端可通过环境变量 `KERAS_BACKEND` 在 `tensorflow` / `jax` / `torch` 之间切换。

> ⚠️ **本机未验证**：我目前只装了 PyTorch，没装 `keras` 包，所以「同一套 Keras 代码换个环境变量就换后端」这个说法来自官方文档，**我没有实测过**。等我装上 keras 并设`KERAS_BACKEND=torch` 跑一次，再回来把这条改成实测结论。
>
> 这一点很关键：如果实测通过，就证明 Keras 确实不依赖 TensorFlow，那我**不需要装那个几百 MB 的 TF 也能跑原书的 Keras 代码**。

### 1.4 本库为什么以 PyTorch 为主线

| 理由 | 说明 |
|:--|:--|
| 论文主流 | 我收藏的深度学习论文实现基本都是 PyTorch，读论文需要对着看 |
| 复试加分 | 复试答「反向传播怎么实现」「为什么用残差连接」时，能手写出训练循环是加分项——PyTorch 逼你手写，Keras 帮你藏起来了 |
| 工业就业 | 岗位需求里 PyTorch 出现频率高于 TF（TF 多在推荐/部署场景） |
| 本书定位 | 原书用 Keras 讲**原理**，用 PyTorch 做**实现**，两者互补而非冲突 |

所以本库的笔记结构是：**原理用 Keras 讲最省事，实现用 PyTorch 练最扎实，API 速查表始终双列。**

## 2. 数学与形式化：张量广播

### 2.1 规则

两个形状要相容，规则是**右对齐、逐维比较**：

| 情况 | 结果 |
|:--|:--|
| 某一维相等 | 直接使用 |
| 其中一个是 `1` | 被「拉伸」成另一个的大小（broadcast） |
| 其余情况 | 报 `RuntimeError` |

实测验证（本机运行结果）：

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
print(a.shape, b.shape)     # torch.Size([3, 1]) torch.Size([1, 4])
print((a + b).shape)        # torch.Size([3, 4])
```

`(3,1)` 与 `(1,4)` 相加得到 `(3,4)`——结果 12 个元素**全都是新算出来的**，不是简单拼接。这个机制的价值在于：

- 省掉了 `repeat` / `tile` 这类显式复制操作，代码短很多
- `(B, N)` 的样本张量能直接和 `(N,)` 的标签向量相加，不用手动加一维
- 深层网络里大量「加法」节点（残差连接、归一化的部分实现）全靠它

### 2.2 符号表

| 符号 | 含义 |
|:--|:--|
| $B$ | batch size |
| $N, C, H, W$ | 序列长度 / 通道 / 高 / 宽 |
| $\odot$ | 逐元素乘（Hadamard 积） |
| $\sigma$ | 激活函数（sigmoid / relu 等） |

### 2.3 广播对应的数学

$a \in \mathbb{R}^{3 \times 1}$，$b \in \mathbb{R}^{1 \times 4}$，结果 $c_{ij} = a_{i1} + b_{1j}$，$c \in \mathbb{R}^{3 \times 4}$。

注意这不是矩阵乘法（内维不同会报错），而是逐元素运算 + 隐式复制。

## 3. 张量形状流

以一个两层 MLP 为例，`B=32`：

| 步骤 | Keras | PyTorch | 输出形状 |
|:--|:--|:--|:--|
| 1 | 输入 | 输入 | `(32, 784)` |
| 2 | `Dense(64, activation="relu")` | `nn.Linear(784, 64)` → `nn.ReLU()` | `(32, 64)` |
| 3 | `Dense(10)` | `nn.Linear(64, 10)` | `(32, 10)` |
| 4 | `CrossEntropyLoss()` | `nn.CrossEntropyLoss()` | 标量 |

**关键点**：PyTorch 的 `nn.Linear` 写作 `nn.Linear(输入维度, 输出维度)`（784 在前），而 Keras 的 `Dense` 写 `Dense(输出维度, activation=...)`。**顺序相反是最容易记混的地方。**

## 4. 最小可运行代码

### 4.1 同一件事的两种写法

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(0)   # 固定种子

# ---------- PyTorch：训练循环手写四步 ----------
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(3):
    for x, y in loader:
        x, y = x.view(-1, 784).to("cuda"), y.to("cuda")
        optimizer.zero_grad()          # ① 清梯度
        loss = loss_fn(model(x), y)
        loss.backward()                # ② 反向传播
        optimizer.step()               # ③ 更新参数
    print(f"epoch {epoch} loss {loss.item():.4f}")
```

```python
# ---------- Keras：训练循环只有一行 ----------
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, epochs=3, batch_size=32)
```

### 4.2 这个差距说明了什么

PyTorch 要写`zero_grad` / `backward` / `step` 三步，Keras 一行 `fit` 全包。

**这不是「谁更高级」的差别，而是设计哲学的差别**：

- Keras 假设你**知道要做什么**，把「怎么做」全包起来
- PyTorch 假设你**知道怎么做**，把控制权交出来

所以文档教程 Keras 更短，而读论文、理解训练细节、调试训练过程时，PyTorch 暴露出的细节让你能看见 Kernels 在干什么。

### 4.3 PyTorch 的后端分三路

```python
print(torch.backends.mps.is_available())   # 输出 False
```

本机实测为 `False`——MPS 是 Apple Silicon 的 GPU 后端，**Windows 上永远是 False**。PyTorch 的后端分`cuda`（NVIDIA）/ `mps`（Apple）/ `cpu` 三路，我的机器只能走 cuda。

## 5. 关键 API 速查

| 概念 | Keras | PyTorch | 备注 |
|:--|:--|:--|:--|
| 建网络 | `keras.Sequential([...])` | `nn.Sequential(...)` | 都支持顺序结构 |
| 稠密层 | `Dense(64, activation="relu")` | `nn.Linear(784, 64)` + `nn.ReLU()` | **维度顺序相反** |
| 卷积层 | `Conv2D(32, 3, activation="relu")` | `nn.Conv2d(3, 32, 3)` + `nn.ReLU()` | **PyTorch 是 (in, out, k)** |
| 展平 | `Flatten()` | `nn.Flatten()` 或 `x.view(B, -1)` | |
| 损失 | `losses.SparseCategoricalCrossentropy()` | `nn.CrossEntropyLoss()` | 注意 logits，见坑 2 |
| 优化器 | `optimizers.Adam(1e-3)` | `torch.optim.Adam(params, lr=1e-3)` | |
| 梯度清零 | 自动 | `optimizer.zero_grad()` | **PyTorch 必写** |
| 反向传播 | 自动 | `loss.backward()` | |
| 参数更新 | 自动 | `optimizer.step()` | |
| 训练 | `model.fit(x, y, epochs=5)` | 手写 for 循环 | 差距最大的一处 |
| 评估 | `model.evaluate(x, y)` | 手写循环 + `argmax` | |
| 打印模型 | `model.summary()` | `print(model)` | |

## 6. 常见坑与排查

### 坑 1：PyTorch 漏写 `optimizer.zero_grad()`

- **症状**：训练能跑，但loss 下降异常缓慢、或前几个 epoch 剧烈抖动，收敛明显比预期差
- **原因**：`backward()` 是**累加**梯度而不是覆盖。不清零的话第 2 个 batch 的梯度里混着第 1 个的，训练变成「把历史梯度全加起来的诡异优化过程」
- **解法**：每个 batch 循环**开头**写 `optimizer.zero_grad()`
- **延伸**：这就是为什么 PyTorch 的标准代码里这三步顺序固定为 `zero_grad` → `backward` → `step`，不能换位置

### 坑 2：`CrossEntropyLoss` 报错或结果不对

- **症状**：`RuntimeError: 1D target tensor expected, multi-target not supported`，或者 loss 数值异常大
- **原因**：`CrossEntropyLoss` 要求喂**原始输出（logits）**，它内部自带 softmax。我在模型末尾习惯性加了一层 `softmax`，结果等价于「softmax 的输出又被softmax 了一次」，梯度被扭曲
- **解法**：`nn.Linear(64, 10)` 后面**不要**加 softmax；想拿概率时用 `torch.softmax(output, dim=1)`，且只用于推理、不参与反传
- **延伸**：**两框架处理方式正好相反**——Keras 用 `from_logits=True/False` 参数来控制是否内部已做softmax；PyTorch 是写死的约定。跨框架迁移代码时这里必踩

### 坑 3：卷积层报维度不匹配

- **症状**：`Given groups=1, weight of size [32, 3, 3, 3], expected input[64, 16, 32, 32] to have 32 channels`
- **原因**：`nn.Conv2d` 的参数顺序是 `(输入通道, 输出通道, 核大小)`，写反了就会把 3 误当成输出通道
- **解法**：逐层打印 `x.shape` 核对每一步；记住 `nn.Conv2d(3, 32, 3)` 表示「吃 3 通道、吐 32 通道」，而 Keras 的 `Conv2D(32, 3)` 是「吐出 32 通道、吃 3 通道」——**两者参数顺序完全相反**
- **延伸**：连接层（Linear / Conv）几乎所有维度报错都源于此，写之前先在纸上确认参数顺序

### 坑 4：JAX 默认不走 GPU

- **症状**：代码能跑，但比纯 CPU 还慢
- **原因**：JAX 默认后端可能不是 GPU，且不像 PyTorch 那样自动探测
- **解法**：显式指定 `jax.config.update("jax_platform_name", "gpu")`；或安装对应的 `jax-cuda` 插件
- **延伸**：框架的「默认行为」各不相同——PyTorch 默认 CUDA，JAX 默认 CPU。**新框架上手第一件事是确认它默认用哪个后端**

### 坑 5：以为装 Keras 就等于装了 TensorFlow

- **症状**：`import keras` 之后找 `keras.backend.tensorflow()`，发现和想的不一样
- **原因**：Keras 3 是独立包，不绑 TF；但很多老教程写的是 `tensorflow.keras`（旧命名空间），那是TF 内置版本
- **解法**：按官方文档用 `import keras`；若教程里是 `from tensorflow import keras`，说明它针对旧版 TF，需要改写
- **延伸**：本机没装 TF 却想跑原书代码，可以装独立 `keras` 包 + `KERAS_BACKEND=torch`——但这条**我还没实测**，见 1.3 节

## 7. 动手练习

### 练习 1：预测广播结果并验证

不用跑，先预测形状，再实际打印确认：

```python
a = torch.randn(32, 1)
b = torch.randn(1, 784)
print((a + b).shape)
```

<details>
<summary>参考答案</summary>

预测：`torch.Size([32, 784])`。

原理：右对齐比较第一维——32 vs 1，1 被拉伸成 32；第二维 1 vs 784，1 被拉伸成 784。结果 32×784=25088 个元素全是新算出来的。

这个形状正是 MLP 里 `(B, 784)` 的输入张量与 `(B, 1)` 的标签向量能直接相加的原因。

</details>

### 练习 2：对比有无 `zero_grad()`

写两个版本的训练循环，各跑 10 个epoch，对比 loss 曲线：

```python
# 版本 A：正确
for x, y in loader:
    optimizer.zero_grad()
    loss_fn(model(x), y).backward()
    optimizer.step()

# 版本 B：故意漏掉第一行
for x, y in loader:
    loss_fn(model(x), y).backward()
    optimizer.step()
```

<details>
<summary>参考答案</summary>

版本 B 的 loss 下降明显更慢，且可能剧烈震荡或直接发散。

因为 `backward()` 累加梯度。第 N 个 batch 的实际梯度是前 N 个 batch 梯度的总和，等效于用了一个越来越大的学习率——这会破坏收敛。

验证方法：在两个循环里都加 `print(loss.item())`，观察版本 B 的 loss 是否单调性差、下降慢。

</details>

## 8. 参考资料

- 《Deep Learning with Python》第 2 版，Ch3「Introduction to TensorFlow, PyTorch, JAX, and Keras」
- [Keras 3 官方文档 · Backend](https://keras.io/api/backend/)
- [PyTorch 官方文档 · Broadcast semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch 官方文档 · Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX 官方文档 · Quickstart](https://docs.jax.dev/en/latest/notebooks/quickstart.html)

## 9. 自检清单

- [ ] 能说出 Keras 和另外三个框架**不在同一层级**
- [ ] 能解释「Keras 是接口层、TensorFlow/PyTorch/JAX 是引擎」
- [ ] 能手写PyTorch 的训练循环四步，并说明漏掉 `zero_grad()` 会怎样
- [ ] 能解释广播规则，并预测 `(32,1)+(1,784)` 的结果形状
- [ ] 记得 `nn.Linear(784,64)` 和 `Dense(64)` 的维度顺序是相反的
- [ ] 知道 `CrossEntropyLoss` 要喂 logits 而不是 softmax 后的概率