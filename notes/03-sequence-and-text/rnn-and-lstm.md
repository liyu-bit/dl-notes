---
chapter: 13
book_section: "Ch13–14 · 序列与文本 · RNN / LSTM 基础"
tags: [RNN, LSTM, 序列模型, 梯度消失, 门控机制]
env: [python 3.12.15, pytorch 2.13.0+cu126]
updated: 2026-11-09
status: draft
---

# RNN 与 LSTM：从循环记忆到门控记忆

> 一句话：搞清 RNN 为什么记不住长距离信息，以及 LSTM 用「一条加法传送带 + 三个门」怎么把它修好。

## 0. TL;DR

- **RNN = 同一单元沿时间复用**：`hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)`，参数量与序列长度无关。
- **梯度消失的根源**是 BPTT 中 Jacobian 连乘 `Wₕᵀ·tanh′`，实际有效记忆约 10~20 步；爆炸用梯度裁剪解决，消失只能改结构。
- **LSTM 的关键是细胞状态**：`Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ`，加法更新使 `∂Cₜ/∂Cₜ₋₁ ≈ fₜ`，梯度可传几百步。
- **三个门各管一件事**：遗忘门删旧、输入门写新、输出门决定对外暴露多少。

## 1. 白话解释

### 1.1 序列数据为什么特殊

MLP 和 CNN 处理的是**相互独立**的样本：一张图片的像素、一个 batch 的样本之间没有时间先后。但现实里大量数据是**序列**——句子的词有前后依赖，股价有趋势，视频帧有连续性。

例如「我今天很____」要填「开心」，模型必须看见前面的上下文。给网络加一个「循环」——**把上一步的隐藏状态传给下一步**——网络就有了短期记忆。这就是 RNN（Elman, 1990）。

但 RNN 的记忆是**每一步都被强制重写一遍**的（tanh 压缩到 -1~1），旧细节会被新信息冲刷。若句子开头的人名要到 50 个词之后才用到，RNN 基本记不住。LSTM（Hochreiter & Schmidhuber, 1997）的动机就是：**让记忆有一条不被重写、只被增删的通道**。

### 1.2 三个东西的一句话直觉

| 模型 | 一句话直觉 |
|:--|:--|
| RNN | 一个只带笔记本封面的读者：每读一个词就把上一页撕掉重写，只留一句"到目前为止的理解" |
| LSTM | 同一个读者，但给他一本可以反复涂改的笔记本 + 三个开关（橡皮擦 / 钢笔 / 嘴） |
| GRU | LSTM 的轻量版：把三个开关合并成两个，参数更少、效果接近 |

**RNN vs LSTM 对比（先看结论，公式在下一节）：**

| 维度 | RNN | LSTM |
|:--|:--|:--|
| 记忆载体 | 只有隐藏状态 `hₜ` | `hₜ`（短期）+ 细胞状态 `Cₜ`（长期） |
| 状态更新 | **乘法重写**：`hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁)` | **加法累积**：`Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ` |
| 信息控制 | 无门控，每步全量覆盖 | 三门自主决定忘 / 记 / 说 |
| 梯度传播 | 连乘 `Wₕᵀ·tanh′`，极易消失或爆炸 | 传送带上近似恒等映射，可传数百步 |
| 参数量 | 少、快 | 约 4 倍，训练更慢 |
| 适用 | 短序列、低算力 | 长依赖：翻译、语音、时序预测 |

### 1.3 典型应用场景

- **RNN**：短文本分类、轻量级时序预测、嵌入式/低算力场景（参数少）。
- **LSTM**：机器翻译与语言模型（Transformer 之前的 SOTA，如 GNMT 2016）、语音识别与合成、时间序列预测（股票 / 交通流 / 负荷）、医疗事件序列建模（电子病历风险预测）、视频行为识别、音乐与手写生成。
- **现状**：NLP 主战场已被 Transformer 取代；但在**流式推理、边缘设备、小数据、强物理时序**场景，LSTM 仍是可靠基线。

## 2. 数学与形式化

### 2.1 RNN：按时间展开的同一单元

![RNN 按时间步展开](https://cdn.jsdelivr.net/gh/liyu-bit/dl-notes@a9152ee63988f0a5205948976bb898324c338985/assets/images/rnn-unrolled.png)

三个时间步共享**同一组参数** `Wₓ、Wₕ、W_y`——这是「循环」的真正含义，也是 RNN 能处理任意长度序列的原因。

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad y_t = \mathrm{softmax}(W_y h_t + b_y)$$

### 2.2 梯度消失 / 爆炸的数学根源

BPTT（沿时间反向传播）把误差从最后一步传回最早一步，链式法则给出：

$$\frac{\partial L}{\partial h_{0}} \propto \prod_{t=1}^{T} \left( W_h^{\top}\,\mathrm{diag}\left[\tanh'(\cdot)\right] \right)$$

因为 `|tanh′| ≤ 1`，若 `Wₕ` 的最大奇异值 `< 1`，连乘几十步后梯度**指数级衰减**（消失）；若 `> 1` 则指数级增长（爆炸）。

| 问题 | 条件 | 后果 | 对策 |
|:--|:--|:--|:--|
| 梯度消失 | `σ_max(Wₕ) < 1` | 远处信息学不到（收不到纠错信号） | 改结构 → LSTM / GRU |
| 梯度爆炸 | `σ_max(Wₕ) > 1` | 损失震荡、参数发散 | `clip_grad_norm_` 裁剪 |

![梯度流对比](https://cdn.jsdelivr.net/gh/liyu-bit/dl-notes@a9152ee63988f0a5205948976bb898324c338985/assets/images/gradient-flow.png)

### 2.3 LSTM：加法传送带 + 三个门

![LSTM 单元结构](https://cdn.jsdelivr.net/gh/liyu-bit/dl-notes@a9152ee63988f0a5205948976bb898324c338985/assets/images/lstm-cell.png)

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{（遗忘门）}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{（输入门）}$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{（候选记忆）}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{（细胞状态更新）}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \qquad h_t = o_t \odot \tanh(C_t) \quad \text{（输出门）}$$

**为什么这样就不过消失？** 沿细胞状态反传时

$$\frac{\partial C_t}{\partial C_{t-k}} = f_t \cdot f_{t-1} \cdots f_{t-k+1} \approx 1 \quad (\text{当 } f \approx 1)$$

路径上只有逐元素乘法**没有矩阵相乘**，且门值是网络学出来的——需要长期保留时它会把 `f` 推到接近 1。这就是「加法通路」比「矩阵乘积通路」更保梯度的本质。

**符号表**

| 符号 | 含义 | 形状 |
|:--|:--|:--|
| `xₜ` | t 时刻输入 | `(B, F)` |
| `hₜ` | 隐藏状态 / 短期记忆 | `(B, H)` |
| `Cₜ` | 细胞状态 / 长期记忆 | `(B, H)` |
| `f, i, o` | 遗忘 / 输入 / 输出门 | `(B, H)`，值域 (0,1) |
| `⊙` | 逐元素乘（Hadamard） | 同形 |

单层 LSTM 参数量约为 `4 × (F·H + H·H + H)`。

## 3. 张量形状流

PyTorch `batch_first=True` 下（Keras 默认就是 batch 在前）：

| 步骤 | 操作 | 输出形状 |
|:--|:--|:--|
| 1 | 输入 `x` | `(B, T, F)` = `(8, 20, 16)` |
| 2 | `nn.LSTM(F→H)` 返回 `out, (hₙ, cₙ)` | `out (8, 20, 32)`；`hₙ/cₙ (1, 8, 32)` |
| 3 | 取最后一步 `out[:, -1]` | `(8, 32)` |
| 4 | `Linear(H→1)` | `(8, 1)` |

要点：`out` 是**每步**的输出（要全序列用这个，等价 Keras 的 `return_sequences=True`）；`hₙ/cₙ` 只是**最后一步**的状态。`num_layers > 1` 或 `bidirectional=True` 时，第一维会变成 `num_layers × 方向数`。

## 4. 最小可运行代码

```python
import torch, torch.nn as nn

torch.manual_seed(0)
B, T, F, H = 8, 20, 16, 32
x = torch.randn(B, T, F)          # (batch, time, feature)


class SeqNet(nn.Module):
    def __init__(self, cell="lstm"):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(F, H, batch_first=True)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        out, _ = self.rnn(x)      # out: (B, T, H)
        return self.head(out[:, -1])   # 等长序列可直接取最后一步


for cell in ("rnn", "lstm"):
    net = SeqNet(cell)
    y = net(x)
    y.sum().backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    print(cell, tuple(y.shape), tuple(net.rnn.weight_hh_l0.shape))
```

Keras 等价写法（三行）：

```python
from tensorflow import keras

model = keras.Sequential([
    keras.Input((T, F)),
    keras.layers.LSTM(H),          # 默认只返回最后一步；return_sequences=True 返回全序列
    keras.layers.Dense(1),
])
```

## 5. 关键 API 速查

| 能力 | Keras | PyTorch |
|:--|:--|:--|
| 基础循环层 | `layers.SimpleRNN(units)` | `nn.RNN(input_size, hidden_size)` |
| 门控层 | `layers.LSTM(units)` / `layers.GRU(units)` | `nn.LSTM(...)` / `nn.GRU(...)` |
| 返回全序列 | `return_sequences=True` | 直接用 `out`（始终返回全序列） |
| 返回最终状态 | `return_state=True` | `out, (hₙ, cₙ) = layer(x)` |
| batch 维位置 | 默认 `(B, T, F)` | 需 `batch_first=True`，否则是 `(T, B, F)` |
| 堆叠多层 | 多个层串联 | `num_layers=n` |
| 双向 | `Bidirectional(layers.LSTM(u))` | `bidirectional=True` |
| 变长序列 | `Masking` / `mask_zero=True` | `pack_padded_sequence` |
| 梯度裁剪 | `clipnorm=1.0` | `clip_grad_norm_(params, 1.0)` |
| 指定初始状态 | `initial_state=[h, c]` | 传入 `(h0, c0)` |

## 6. 常见坑与排查

### 坑 1：形状对不上，报 "input must have 3 dimensions"

- **症状**：`RuntimeError: input must have 3 dimensions, got 2`，或训练效果莫名很差但 loss 正常。
- **原因**：PyTorch 默认 `batch_first=False`，输入被当作 `(T, B, F)`。你把 `(B, T, F)` 喂进去，模型把 batch 当成了时间步。
- **解法**：显式 `nn.LSTM(F, H, batch_first=True)`，并在 forward 里断言 `assert x.dim() == 3`。
- **延伸**：LSTM 层堆叠时如果只给第一层设了 `batch_first`，后面接 `Linear` 取 `out[:, -1]` 会静默取错维度——症状是 loss 能降但准确率不涨。

### 坑 2：变长序列取最后一步，取到了 padding

- **症状**：验证集指标明显低于训练集，且随 padding 比例升高而恶化。
- **原因**：`out[:, -1]` 取的是**补零后的最后一位**，不是真实序列末尾；pad 位置的隐状态没有意义。
- **解法**：用长度索引 `out[torch.arange(B), lengths - 1]`，或改用 `pack_padded_sequence`。

### 坑 3：训练 loss 变 NaN

- **症状**：几个 step 后 loss 直接 `nan`。
- **原因**：梯度爆炸（BPTT 连乘放大），学习率偏大时更常见。
- **解法**：`nn.utils.clip_grad_norm_(model.parameters(), 1.0)` 放在 `backward()` 之后、`step()` 之前；必要时把 lr 降到 1e-3 以下。

### 坑 4：以为 LSTM 就完全不梯度消失了

- **症状**：层数加深或序列上千步后，LSTM 也学不动远处依赖。
- **原因**：细胞状态通路近似恒等，但**输出门和隐藏状态通路**仍含矩阵乘；层与层之间也没有豁免。
- **解法**：需要极长依赖时用注意力 / Transformer；或减小层数、加残差、用 LayerNorm。

## 7. 动手练习

1. 用上面的 `SeqNet` 做一个「记住第一个 token」的任务：序列长度 100，标签 = 第一个 token 的符号。分别用 `rnn` 和 `lstm` 训练 200 步，比较两者的准确率。
2. 把 LSTM 的遗忘门偏置 `b_f` 初始化为 1.0（其余不变），观察收敛速度变化，并解释原因。

<details>
<summary>参考答案</summary>

1. RNN 通常停在 50%~60%（接近随机猜测），LSTM 能到 90% 以上。原因：100 步远超 RNN 的有效记忆长度，梯度传不回去；LSTM 的细胞状态通路保住了梯度。
2. 收敛更快。`b_f = 1` 使初始 `f ≈ σ(1) ≈ 0.73`，倾向于「先记住」而不是「先遗忘」，缓解训练初期信息过快流失（这是 `forget_bias=1.0` 的经验做法，TF/Keras 默认就这样初始化）。

</details>

## 8. 参考资料

- 《Deep Learning with Python》第 2 版，Ch13 时间序列预测 / Ch14 文本分类（LSTM 主线示例）。
- Elman, J. L. (1990). *Finding structure in time.* Cognitive Science, 14(2), 179–211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder*（GRU 提出）.
- PyTorch 官方文档：`torch.nn.LSTM` / `torch.nn.utils.clip_grad_norm_`。
- 延伸：Bahdanau et al. (2014) 注意力机制 → Vaswani et al. (2017) Transformer（理解「为什么 RNN 被替代」的必读链）。

## 9. 自检清单

- [ ] 能用一句话解释 RNN 的「循环」到底循环的是什么（参数复用 + 状态传递）
- [ ] 能写出 RNN 与 LSTM 的核心公式，并说明 `Cₜ` 与 `hₜ` 的分工
- [ ] 能从链式法则说明梯度消失的成因，并解释 LSTM 为什么能缓解
- [ ] 能写出最小可运行代码，并说清 `out` 与 `hₙ` 的区别
- [ ] 能说出至少两个常见坑（batch_first 顺序、padding 取错最后一步）
