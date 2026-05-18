"""
Device manager module.

This module checks the local running environment for the project.

Main functions:
1. Check Python version
2. Check PyTorch version
3. Check whether Intel XPU is available
4. Select the best available device
5. Print a readable device report

This project is designed for local LLM / RAG experiments on Intel Core Ultra
devices with PyTorch XPU support.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class DeviceInfo:
    """
    Store basic device information.

    Attributes
    ----------
    python_version:
        Current Python version.
    platform:
        Operating system and architecture information.
    torch_version:
        Current PyTorch version.
    xpu_available:
        Whether PyTorch can detect Intel XPU.
    xpu_count:
        Number of available XPU devices.
    selected_device:
        Device selected for model inference.
    xpu_device_name:
        Name of the XPU device if available.
    cpu_info:
        CPU information from system command.
    memory_info:
        Memory information from system command.
    """

    python_version: str
    platform: str
    torch_version: str
    xpu_available: bool
    xpu_count: int
    selected_device: str
    xpu_device_name: Optional[str]
    cpu_info: str
    memory_info: str


def run_command(command: list[str]) -> str:
    """
    Run a system command safely.

    If the command fails, return a readable error message instead of crashing.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return f"Command failed: {' '.join(command)}"

        return result.stdout.strip()

    except Exception as error:
        return f"Command error: {error}"


def get_cpu_info() -> str:
    """
    Get CPU information.

    On Linux, this function tries to use lscpu.
    If lscpu is unavailable, it falls back to platform.processor().
    """
    cpu_info = run_command(["lscpu"])

    if cpu_info.startswith("Command failed") or cpu_info.startswith("Command error"):
        return platform.processor() or "Unknown CPU"

    lines = cpu_info.splitlines()

    useful_lines = []

    for line in lines:
        if any(
            key in line
            for key in [
                "Model name",
                "型号名称",
                "CPU(s)",
                "CPU:",
                "Thread",
                "每个核的线程数",
                "Core",
                "每个座的核数",
            ]
        ):
            useful_lines.append(line)

    return "\n".join(useful_lines) if useful_lines else cpu_info


def get_memory_info() -> str:
    """
    Get memory information.

    On Linux, this function uses free -h.
    """
    memory_info = run_command(["free", "-h"])

    if memory_info.startswith("Command failed") or memory_info.startswith("Command error"):
        return "Unknown memory information"

    return memory_info


def get_xpu_device_name(index: int = 0) -> Optional[str]:
    """
    Get Intel XPU device name.

    PyTorch XPU may not expose the same device name API on all versions,
    so this function uses a safe try-except block.
    """
    if not torch.xpu.is_available():
        return None

    try:
        return torch.xpu.get_device_name(index)
    except Exception:
        return f"Intel XPU device {index}"


def select_best_device() -> str:
    """
    Select the best available device.

    Priority:
    1. Intel XPU
    2. CPU
    """
    if torch.xpu.is_available():
        return "xpu"

    return "cpu"


def get_device_info() -> DeviceInfo:
    """
    Collect all device information.
    """
    xpu_available = torch.xpu.is_available()
    xpu_count = torch.xpu.device_count() if xpu_available else 0

    selected_device = select_best_device()

    return DeviceInfo(
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        torch_version=torch.__version__,
        xpu_available=xpu_available,
        xpu_count=xpu_count,
        selected_device=selected_device,
        xpu_device_name=get_xpu_device_name(0) if xpu_available else None,
        cpu_info=get_cpu_info(),
        memory_info=get_memory_info(),
    )


def get_device_info_dict() -> Dict[str, object]:
    """
    Return device information as a dictionary.

    This is useful for Streamlit pages or JSON output.
    """
    info = get_device_info()

    return {
        "python_version": info.python_version,
        "platform": info.platform,
        "torch_version": info.torch_version,
        "xpu_available": info.xpu_available,
        "xpu_count": info.xpu_count,
        "selected_device": info.selected_device,
        "xpu_device_name": info.xpu_device_name,
        "cpu_info": info.cpu_info,
        "memory_info": info.memory_info,
    }


def print_device_report() -> None:
    """
    Print a readable device report in terminal.
    """
    info = get_device_info()

    print("=" * 80)
    print("Local Device Report")
    print("=" * 80)

    print(f"Python version: {info.python_version}")
    print(f"Platform: {info.platform}")
    print(f"PyTorch version: {info.torch_version}")

    print("-" * 80)
    print("XPU Status")
    print("-" * 80)
    print(f"XPU available: {info.xpu_available}")
    print(f"XPU count: {info.xpu_count}")
    print(f"XPU device name: {info.xpu_device_name}")
    print(f"Selected device: {info.selected_device}")

    print("-" * 80)
    print("CPU Info")
    print("-" * 80)
    print(info.cpu_info)

    print("-" * 80)
    print("Memory Info")
    print("-" * 80)
    print(info.memory_info)

    print("=" * 80)

    if info.selected_device == "xpu":
        print("Result: Intel XPU is available. The project can use PyTorch XPU.")
    else:
        print("Result: XPU is not available. The project will use CPU.")


def main() -> None:
    """
    Run device check from command line.
    """
    print_device_report()


if __name__ == "__main__":
    main()