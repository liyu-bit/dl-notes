# notes · 笔记目录

每篇笔记遵循统一的**九段式**结构（见 [`_template.md`](_template.md)）。
新增笔记请用脚本生成骨架：

```bash
python ../scripts/new_note.py --stage 01-foundations --title "神经网络数学构件" --chapter 2
```

## 阶段与章节对照

| 阶段 | 目录 | 章节 | 说明 |
|:--|:--|:--|:--|
| S0 | [`00-setup/`](00-setup/) | Ch3 | 环境、框架对比、CUDA 排错 |
| S1 | [`01-foundations/`](01-foundations/) | Ch1/2/4/5/6/7 | 基础与通用工作流 |
| S2 | [`02-cnn/`](02-cnn/) | Ch8–12 | 视觉 |
| S3 | [`03-sequence-and-text/`](03-sequence-and-text/) | Ch13–16 | 序列与文本 |
| S4 | [`04-generation/`](04-generation/) | Ch17 | 生成 |
| S4 | [`05-practice/`](05-practice/) | Ch18–20 | 工程实践与读后感 |

## 写作约定

- **一文件一主题**，单文件不超过 800 行，超了按节拆成 `xxx-part2.md`
- **公式用 LaTeX**：`$...$` 行内，`$$...$$` 独立成行
- **代码块必须标语言**：` ```python `、` ```bash `、` ```yaml `
- **图片放 `../assets/images/`**，用相对路径引用，保证 GitHub 与本地都能渲染
- **中英术语**：首次出现写「中文（English）」，之后统一用中文
- **状态标记**：`status: draft` → `reviewed`，根 README 的索引表同步更新
