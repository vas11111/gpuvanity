"""
Quick single-GPU benchmark.

Usage:
    python3 bench.py              # auto-detect first GPU, 10 iterations
    python3 bench.py --iters 20   # 20 iterations
    python3 bench.py --gpu 2      # use GPU index 2
"""
import multiprocessing
import time
import warnings

import numpy as np
import pyopencl as cl

warnings.filterwarnings("ignore", category=cl.CompilerWarning)

from core.devices import discover_gpus
from core.program import build_program_source
from core.workload import WorkloadConfig


def run_bench(gpu_idx: int = 0, iters: int = 10, batch_exp: int = 28):
    gpus = discover_gpus()
    if not gpus:
        print("No GPUs found.")
        return
    if gpu_idx >= len(gpus):
        print(f"GPU {gpu_idx} not found ({len(gpus)} available).")
        return

    hw = gpus[gpu_idx]
    vram = hw.global_mem_size / (1 << 30)
    print(f"GPU: {hw.name}  ({hw.max_compute_units} CUs, {vram:.1f} GB)")
    print(f"Batch: 2^{batch_exp} = {1 << batch_exp:,} keys/iter")
    print(f"Iterations: {iters}")
    print()

    src = build_program_source(("Bench",), (), True)
    cfg = WorkloadConfig(src, batch_exp)

    ctx = cl.Context([hw])
    try:
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE,
        )
    except cl.RuntimeError:
        queue = cl.CommandQueue(ctx)

    flags = "-cl-fast-relaxed-math -cl-mad-enable"
    print("Compiling kernel... ", end="", flush=True)
    t_compile = time.monotonic()
    binary = cl.Program(ctx, src).build(options=flags)
    kern = cl.Kernel(binary, "ed25519_scan")
    print(f"done ({time.monotonic() - t_compile:.1f}s)")

    max_wg = kern.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, hw)
    preferred = kern.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, hw,
    )
    local_dim = min(preferred * (max_wg // preferred), max_wg)
    global_dim = ((cfg.batch_size + local_dim - 1) // local_dim) * local_dim

    MF = cl.mem_flags
    buf_seed = cl.Buffer(ctx, MF.READ_ONLY | MF.ALLOC_HOST_PTR, 32)
    buf_result = cl.Buffer(ctx, MF.WRITE_ONLY | MF.ALLOC_HOST_PTR, 33)
    buf_sweep = cl.Buffer(
        ctx, MF.READ_ONLY | MF.COPY_HOST_PTR,
        hostbuf=np.array([cfg.sweep_bytes], dtype=np.uint8),
    )
    buf_rank = cl.Buffer(
        ctx, MF.READ_ONLY | MF.COPY_HOST_PTR,
        hostbuf=np.array([0], dtype=np.uint8),
    )
    result_host = np.zeros(33, dtype=np.uint8)

    kern.set_arg(0, buf_seed)
    kern.set_arg(1, buf_result)
    kern.set_arg(2, buf_sweep)
    kern.set_arg(3, buf_rank)

    # warmup (1 iteration, not timed)
    cl.enqueue_copy(queue, buf_seed, cfg.seed, is_blocking=False)
    cl.enqueue_nd_range_kernel(queue, kern, (global_dim,), (local_dim,))
    cl.enqueue_copy(queue, result_host, buf_result, is_blocking=True)
    cfg.step()

    print()
    total_keys = 0
    rates = []
    t_total = time.monotonic()

    for i in range(iters):
        cl.enqueue_copy(queue, buf_seed, cfg.seed, is_blocking=False)
        t0 = time.monotonic()
        cl.enqueue_nd_range_kernel(queue, kern, (global_dim,), (local_dim,))
        cl.enqueue_copy(queue, result_host, buf_result, is_blocking=True)
        dt = time.monotonic() - t0
        cfg.step()

        keys = global_dim
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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import argparse
    p = argparse.ArgumentParser(description="Single-GPU throughput benchmark")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    p.add_argument("--iters", type=int, default=10, help="Benchmark iterations (default: 10)")
    p.add_argument("--batch-exp", type=int, default=28, help="Batch exponent (default: 28)")
    args = p.parse_args()

    run_bench(gpu_idx=args.gpu, iters=args.iters, batch_exp=args.batch_exp)
