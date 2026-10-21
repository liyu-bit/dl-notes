#!/usr/bin/env python
"""环境自检：打印 Python / PyTorch / CUDA / 显卡 / 依赖版本矩阵。

用法：
    python scripts/check_env.py
    python scripts/check_env.py --json      # 机器可读输出

退出码：0 = GPU 可用（或明确为纯 CPU 环境）；1 = 装了 torch 但 CUDA 不可用（需排查）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys

PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "matplotlib",
    "seaborn",
    "yaml",
    "tqdm",
    "tensorboard",
]


def _version(mod_name: str) -> str | None:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def collect() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }

    for name in PACKAGES:
        info["packages"][name] = _version(name)

    try:
        import torch
    except Exception:
        info["torch_installed"] = False
        return info

    info["torch_installed"] = True
    info["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    info["cuda_available"] = bool(torch.cuda.is_available())

    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            free, total = torch.cuda.mem_get_info()
            info["gpu_mem_total_gb"] = round(total / 1024**3, 2)
            info["gpu_mem_free_gb"] = round(free / 1024**3, 2)
        except Exception:
            pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="深度学习环境自检")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    info = collect()

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0 if info.get("torch_installed") and info.get("cuda_available") else 1

    print("=" * 56)
    print("深度学习环境自检")
    print("=" * 56)
    print(f"Python      : {info['python']}  ({info['executable']})")
    print(f"系统        : {info['platform']} [{info['machine']}]")
    print("-" * 56)

    print("依赖版本：")
    for name, ver in info["packages"].items():
        mark = "OK " if ver else "-- "
        print(f"  [{mark}] {name:<14} {ver or '未安装'}")

    print("-" * 56)
    if not info["torch_installed"]:
        print("PyTorch 未安装。")
        print("  CPU: pip install -r requirements.txt")
        print("  GPU: pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu124")
        return 1

    print(f"PyTorch CUDA 构建版本 : {info['torch_cuda_build'] or 'CPU-only'}")
    print(f"CUDA 可用            : {'是' if info['cuda_available'] else '否'}")

    if info["cuda_available"]:
        print(f"设备                 : {info['device_name']}  (sm_{info.get('capability', '?')})")
        if "gpu_mem_total_gb" in info:
            print(f"显存                 : 总 {info['gpu_mem_total_gb']} GB / 空闲 {info['gpu_mem_free_gb']} GB")
        print("\n环境就绪，可以开始跑示例。")
        return 0

    print("\n[警告] 装了 PyTorch 但 CUDA 不可用，按顺序排查：")
    print("  1. nvidia-smi —— 确认驱动正常、且驱动版本 >= 所需 CUDA 版本")
    print("  2. pip list | grep -i torch —— 确认没有 conda/pip 混装两套 torch")
    print("  3. 笔记本 GPU 需 sm_89 架构，用官方 cu124 轮子重装")
    print("  4. 重装：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
