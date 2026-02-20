import logging
import os
import sys
from typing import List, Tuple

import click
import pycuda.driver as cuda

os.environ.setdefault("CUDA_CACHE_DISABLE", "0")


def _ensure_init():
    try:
        cuda.Device.count()
    except cuda.LogicError:
        cuda.init()


def discover_gpus() -> List[int]:
    """Return device indices for every CUDA GPU, sorted by SM count descending."""
    _ensure_init()
    n = cuda.Device.count()
    devs = list(range(n))
    devs.sort(
        key=lambda i: cuda.Device(i).get_attribute(
            cuda.device_attribute.MULTIPROCESSOR_COUNT
        ),
        reverse=True,
    )
    return devs


def get_device_info(idx: int) -> dict:
    _ensure_init()
    dev = cuda.Device(idx)
    return {
        "name": dev.name(),
        "sms": dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
        "vram_gb": dev.total_memory() / (1 << 30),
    }


def select_gpus(device_indices: List[int]) -> List[int]:
    _ensure_init()
    n = cuda.Device.count()
    for i in device_indices:
        if i >= n:
            raise ValueError(f"GPU {i} not found ({n} available)")
    return device_indices


def pick_devices() -> List[int]:
    """Resolve GPU selection from env or interactive prompt."""
    _ensure_init()
    env = os.environ.get("VANITY_DEVICES")
    if env:
        return [int(x) for x in env.split(",")]

    n = cuda.Device.count()
    if n == 0:
        logging.error("No CUDA GPUs found")
        sys.exit(1)

    click.echo("GPU(s):")
    for i in range(n):
        info = get_device_info(i)
        click.echo(
            f"  [{i}] {info['name']}  --  {info['sms']} SMs, {info['vram_gb']:.1f} GB"
        )

    all_ids = ",".join(str(i) for i in range(n))
    raw = click.prompt("Choose (comma-separated)", default=all_ids)
    ids = [int(x) for x in raw.split(",")]

    click.echo(f"Set VANITY_DEVICES='{raw}' to skip this next time.")
    return ids
