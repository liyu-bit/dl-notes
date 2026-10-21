#!/usr/bin/env python
"""按九段式模板生成新笔记骨架。

用法：
    python scripts/new_note.py --stage 01-foundations --title "神经网络数学构件"
    python scripts/new_note.py --stage 02-cnn --title "图像分类" --chapter 8
    python scripts/new_note.py --list-stages

会生成：notes/<stage>/<slug>.md
若目标文件已存在则拒绝覆盖（避免误删已写好的内容）。
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = REPO_ROOT / "notes"

STAGES = {
    "00-setup": "环境与工具",
    "01-foundations": "基础与通用工作流",
    "02-cnn": "视觉：CNN",
    "03-sequence-and-text": "序列与文本",
    "04-generation": "生成与进阶",
    "05-practice": "实践与工程",
}


def slugify(title: str) -> str:
    """中文标题转拼音不现实，退化为「取英文/数字部分 + 短哈希」由用户手改。"""
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    if ascii_part and len(ascii_part) > 2:
        return ascii_part
    # 纯中文标题：给个可读的占位名，提示手动改名
    return "note-" + str(abs(hash(title)) % 100000).zfill(5)


TEMPLATE = """---
chapter: {chapter}
book_section: "{book_section}"
tags: []
env: []
updated: {today}
status: draft
---

# {title}

> 一句话说明这篇笔记解决什么问题。

## 0. TL;DR

- 三行以内，能量出「是否真的懂了」。

## 1. 白话解释

用生活化类比 + 一句「它解决了什么问题」。不要抄书上的定义。

## 2. 数学与形式化

核心公式（LaTeX），附符号表与维度约束。不堆推导细节。

## 3. 张量形状流

| 步骤 | 操作 | 输出形状 |
|:--|:--|:--|
| 1 | 输入 | `(B, 32, 32)` |
| 2 | Conv2d(3→32, 3) | `(B, 32, 30, 30)` |

## 4. 最小可运行代码

不超过 30 行，复制粘贴能跑，含随机种子。

```python
# TODO
```

## 5. 关键 API 速查

| API | Keras | PyTorch | 备注 |
|:--|:--|:--|:--|
| TODO | | | |

## 6. 常见坑与排查

### 坑 1：<一句话症状>

- **症状**：报错原文 / 现象
- **原因**：根本原因
- **解法**：可直接复制的修复代码
- **延伸**：什么时候还会遇到

## 7. 动手练习

1. 原创小题（10 分钟内可验证）

<details>
<summary>参考答案</summary>

答案

</details>

## 8. 参考资料

- 《Deep Learning with Python》第 2 版，对应章节
- 官方文档（永久链接）
- 延伸论文（arXiv / DOI）

## 9. 自检清单

- [ ] 能用一句话解释这个概念
- [ ] 能推导核心公式
- [ ] 能写出最小可运行代码
- [ ] 能说出至少一个常见坑
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成新笔记骨架")
    parser.add_argument("--stage", choices=sorted(STAGES), help="所属阶段")
    parser.add_argument("--title", help="笔记标题")
    parser.add_argument("--chapter", type=int, default=None, help="对应书中章节号")
    parser.add_argument("--slug", help="自定义文件名（不含 .md），纯中文标题时建议指定英文名")
    parser.add_argument("--list-stages", action="store_true", help="列出所有阶段")
    args = parser.parse_args()

    if args.list_stages:
        print("可用阶段：")
        for key, desc in STAGES.items():
            print(f"  {key:<24} {desc}")
        return 0

    if not args.stage or not args.title:
        parser.print_help()
        print("\n错误：需要 --stage 和 --title（或使用 --list-stages）", file=sys.stderr)
        return 1

    slug = args.slug or slugify(args.title)
    target_dir = NOTES_DIR / args.stage
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{slug}.md"

    if target.exists():
        print(f"[错误] 文件已存在，拒绝覆盖：{target}", file=sys.stderr)
        return 1

    content = TEMPLATE.format(
        chapter=args.chapter if args.chapter is not None else "null",
        book_section=f"Ch{args.chapter} · {args.title}" if args.chapter else "",
        title=args.title,
        today=datetime.date.today().isoformat(),
    )
    target.write_text(content, encoding="utf-8")

    print(f"[完成] 已生成：{target.relative_to(REPO_ROOT)}")
    if not args.slug and re.search(r"[\u4e00-\u9fff]", args.title):
        print(f"[提示] 标题含中文，文件名自动生成的是占位名，建议手动重命名为英文 slug")
    print("下一步：填内容 → 更新根 README 的目录索引状态 → 提交")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
