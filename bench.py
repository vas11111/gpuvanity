"""
Quick single-GPU benchmark (CUDA).

Usage:
    python3 bench.py              # auto-detect first GPU, 10 iterations
    python3 bench.py --iters 20   # 20 iterations
    python3 bench.py --gpu 2      # use GPU index 2
"""
import multiprocessing
import time

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from core.devices import discover_gpus, get_device_info
from core.program import build_program_source
from core.workload import WorkloadConfig


def run_bench(gpu_idx: int = 0, iters: int = 10, batch_exp: int = 28):
    cuda.init()

    n = cuda.Device.count()
    if n == 0:
        print("No GPUs found.")
        return
    if gpu_idx >= n:
        print(f"GPU {gpu_idx} not found ({n} available).")
        return

    device = cuda.Device(gpu_idx)
    info = get_device_info(gpu_idx)
    print(f"GPU: {info['name']}  ({info['sms']} SMs, {info['vram_gb']:.1f} GB)")
    print(f"Batch: 2^{batch_exp} = {1 << batch_exp:,} keys/iter")
    print(f"Iterations: {iters}")
    print()

    ctx = device.make_context()

    src = build_program_source(("Bench",), (), True)
    cfg = WorkloadConfig(src, batch_exp)

    print("Compiling kernel... ", end="", flush=True)
    t_compile = time.monotonic()
    mod = SourceModule(
        src,
        options=["--use_fast_math", "-O3", "--ptxas-options=-O3"],
        no_extern_c=True,
    )
    kern = mod.get_function("ed25519_scan")
    print(f"done ({time.monotonic() - t_compile:.1f}s)")

    block_dim = 256
    grid_dim = (cfg.batch_size + block_dim - 1) // block_dim

    d_seed = cuda.mem_alloc(32)
    d_result = cuda.mem_alloc(33)
    d_sweep = cuda.mem_alloc(1)
    d_rank = cuda.mem_alloc(1)

    cuda.memcpy_htod(d_sweep, np.array([cfg.sweep_bytes], dtype=np.uint8))
    cuda.memcpy_htod(d_rank, np.array([0], dtype=np.uint8))
    result_host = np.zeros(33, dtype=np.uint8)

    # warmup
    cuda.memcpy_htod(d_seed, cfg.seed)
    kern(d_seed, d_result, d_sweep, d_rank, block=(block_dim, 1, 1), grid=(grid_dim, 1))
    cuda.memcpy_dtoh(result_host, d_result)
    cfg.step()

    print()
    total_keys = 0
    rates = []
    t_total = time.monotonic()

    for i in range(iters):
        cuda.memcpy_htod(d_seed, cfg.seed)
        t0 = time.monotonic()
        kern(d_seed, d_result, d_sweep, d_rank, block=(block_dim, 1, 1), grid=(grid_dim, 1))
        cuda.memcpy_dtoh(result_host, d_result)
        dt = time.monotonic() - t0
        cfg.step()

        keys = block_dim * grid_dim
        total_keys += keys
        rate = keys / (dt * 1e6)
        rates.append(rate)
        print(f"  iter {i+1:3d}/{iters}:  {rate:7.2f} MH/s  ({dt*1000:.0f}ms)")

    wall = time.monotonic() - t_total
    avg = total_keys / (wall * 1e6)
    peak = max(rates)
    low = min(rates)

    print()
    print(f"{'='*44}")
    print(f"  Iterations:  {iters}")
    print(f"  Total keys:  {total_keys:,}")
    print(f"  Wall time:   {wall:.2f}s")
    print(f"  Average:     {avg:.2f} MH/s")
    print(f"  Peak:        {peak:.2f} MH/s")
    print(f"  Low:         {low:.2f} MH/s")
    print(f"{'='*44}")

    ctx.pop()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import argparse
    p = argparse.ArgumentParser(description="Single-GPU throughput benchmark")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    p.add_argument("--iters", type=int, default=10, help="Benchmark iterations (default: 10)")
    p.add_argument("--batch-exp", type=int, default=28, help="Batch exponent (default: 28)")
    args = p.parse_args()

    run_bench(gpu_idx=args.gpu, iters=args.iters, batch_exp=args.batch_exp)
