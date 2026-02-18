import logging
import os
import sys
from typing import List, Tuple

import click
import pyopencl as cl

os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
if os.environ.get("DEBUG"):
    os.environ["PYOPENCL_NO_CACHE"] = "TRUE"


def discover_gpus() -> List[cl.Device]:
    """Find every GPU across all OpenCL platforms, strongest first."""
    found = [
        d
        for plat in cl.get_platforms()
        for d in plat.get_devices(device_type=cl.device_type.GPU)
    ]
    found.sort(key=lambda d: d.max_compute_units, reverse=True)
    return found


def select_gpus(platform_idx: int, device_indices: List[int]) -> List[cl.Device]:
    """Return specific GPUs from a specific platform."""
    all_devs = cl.get_platforms()[platform_idx].get_devices(device_type=cl.device_type.GPU)
    return [all_devs[i] for i in device_indices]


def pick_devices() -> Tuple[int, List[int]]:
    """Resolve GPU selection from env or interactive prompt."""
    env = os.environ.get("VANITY_DEVICES")
    if env:
        plat_str, dev_str = env.split(":")
        return int(plat_str), [int(x) for x in dev_str.split(",")]

    platforms = cl.get_platforms()
    click.echo("Platform:")
    for i, p in enumerate(platforms):
        click.echo(f"  [{i}] {p.name}")

    plat = click.prompt("Choose", default=0, type=click.IntRange(0, len(platforms) - 1))
    devs = platforms[plat].get_devices(device_type=cl.device_type.GPU)

    if not devs:
        logging.error(f"Platform {plat} has no GPUs")
        sys.exit(1)

    click.echo("GPU(s):")
    for i, d in enumerate(devs):
        vram = d.global_mem_size / (1 << 30)
        click.echo(f"  [{i}] {d.name}  --  {d.max_compute_units} CUs, {vram:.1f} GB")

    all_ids = ",".join(str(i) for i in range(len(devs)))
    raw = click.prompt("Choose (comma-separated)", default=all_ids)
    ids = [int(x) for x in raw.split(",")]

    click.echo(f"Set VANITY_DEVICES='{plat}:{raw}' to skip this next time.")
    return plat, ids
