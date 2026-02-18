import logging
import os
import time
import warnings
from ctypes import c_uint64
from multiprocessing import Array, Queue
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

warnings.filterwarnings("ignore", category=cl.CompilerWarning)

from core.workload import WorkloadConfig
from core.devices import discover_gpus, select_gpus

THROUGHPUT_LOG_INTERVAL = 5.0
_TICK_CHECK_INTERVAL = 8


def _fmt_count(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


class GPUMiner:
    """Binds one OpenCL device and searches continuously until stopped."""

    __slots__ = (
        "ctx", "cmd_queue", "kern", "cfg",
        "rank", "label",
        "buf_seed", "buf_result", "buf_sweep_len", "buf_rank",
        "result_host", "global_dim", "local_dim",
        "_interval_keys", "_lifetime_keys", "_last_report", "_tick_count",
    )

    def __init__(
        self,
        program_src: str,
        rank: int,
        cfg: WorkloadConfig,
        device_selection: Optional[Tuple[int, List[int]]] = None,
    ):
        gpus = (
            discover_gpus()
            if device_selection is None
            else select_gpus(*device_selection)
        )
        hw = gpus[rank]
        self.ctx = cl.Context([hw])
        self.cfg = cfg
        self.rank = rank
        self.label = rank if device_selection is None else device_selection[1][rank]

        try:
            self.cmd_queue = cl.CommandQueue(
                self.ctx,
                properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE,
            )
        except cl.RuntimeError:
            self.cmd_queue = cl.CommandQueue(self.ctx)

        flags = "-cl-fast-relaxed-math -cl-mad-enable"

        logging.info(f"GPU {rank}: compiling kernel (first run may take 15-30s)...")
        binary = cl.Program(self.ctx, program_src).build(options=flags)
        self.kern = cl.Kernel(binary, "ed25519_scan")
        logging.info(f"GPU {rank}: kernel ready")

        max_wg = self.kern.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, hw,
        )
        preferred = self.kern.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, hw,
        )
        self.local_dim = min(preferred * (max_wg // preferred), max_wg)
        self.global_dim = (
            (cfg.batch_size + self.local_dim - 1) // self.local_dim
        ) * self.local_dim

        cfg._stride = self.global_dim

        self._setup_buffers()
        self._interval_keys = 0
        self._lifetime_keys = 0
        self._last_report = time.monotonic()
        self._tick_count = 0

    def _setup_buffers(self) -> None:
        MF = cl.mem_flags
        self.buf_seed = cl.Buffer(self.ctx, MF.READ_ONLY | MF.ALLOC_HOST_PTR, 32)
        self.buf_result = cl.Buffer(self.ctx, MF.WRITE_ONLY | MF.ALLOC_HOST_PTR, 33)
        self.buf_sweep_len = cl.Buffer(
            self.ctx, MF.READ_ONLY | MF.COPY_HOST_PTR,
            hostbuf=np.array([self.cfg.sweep_bytes], dtype=np.uint8),
        )
        self.buf_rank = cl.Buffer(
            self.ctx, MF.READ_ONLY | MF.COPY_HOST_PTR,
            hostbuf=np.array([0], dtype=np.uint8),
        )
        self.result_host = np.zeros(33, dtype=np.uint8)
        self.kern.set_arg(0, self.buf_seed)
        self.kern.set_arg(1, self.buf_result)
        self.kern.set_arg(2, self.buf_sweep_len)
        self.kern.set_arg(3, self.buf_rank)

    def tick(self) -> np.ndarray:
        """Run one batch on the GPU and return the 33-byte result buffer."""
        cl.enqueue_copy(self.cmd_queue, self.buf_seed, self.cfg.seed, is_blocking=False)

        cl.enqueue_nd_range_kernel(
            self.cmd_queue, self.kern,
            (self.global_dim,), (self.local_dim,),
        )

        self.cfg.step()

        cl.enqueue_copy(self.cmd_queue, self.result_host, self.buf_result, is_blocking=True)

        self._interval_keys += self.global_dim
        self._lifetime_keys += self.global_dim
        self._tick_count += 1

        if self._tick_count >= _TICK_CHECK_INTERVAL:
            self._tick_count = 0
            now = time.monotonic()
            dt = now - self._last_report
            if dt >= THROUGHPUT_LOG_INTERVAL:
                rate = self._interval_keys / (dt * 1e6)
                logging.info(
                    f"GPU {self.label}: {rate:.2f} MH/s | "
                    f"{_fmt_count(self._lifetime_keys)} searched"
                )
                self._interval_keys = 0
                self._last_report = now

        return self.result_host


def mine_loop(
    rank: int,
    cfg: WorkloadConfig,
    halt: Synchronized,
    hits: Queue,
    counters: Array,
    device_selection: Optional[Tuple[int, List[int]]] = None,
) -> None:
    """Long-lived process: mine on one GPU, push found keys to `hits` queue."""
    try:
        try:
            os.sched_setaffinity(0, {rank % (os.cpu_count() or 1)})
        except (AttributeError, OSError):
            pass

        miner = GPUMiner(
            program_src=cfg.program_src,
            rank=rank,
            cfg=cfg,
            device_selection=device_selection,
        )

        while not halt.value:
            result = miner.tick()

            counters[rank] = miner._lifetime_keys

            if result[0]:
                hits.put(bytes(result[1:33]))

                miner.result_host[:] = 0
                cl.enqueue_copy(
                    miner.cmd_queue, miner.buf_result,
                    miner.result_host, is_blocking=True,
                )

                miner.cfg.randomize()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.exception(f"GPU {rank} crashed: {exc}")
