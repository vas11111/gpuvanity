import logging
import os
import time
from multiprocessing import Array, Queue
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from core.workload import WorkloadConfig
from core.devices import discover_gpus

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
    """Binds one CUDA device and searches continuously until stopped."""

    __slots__ = (
        "kern", "cfg",
        "rank", "label",
        "d_seed", "d_result", "d_sweep_len", "d_rank",
        "result_host", "block_dim", "grid_dim",
        "_interval_keys", "_lifetime_keys", "_last_report", "_tick_count",
    )

    def __init__(
        self,
        program_src: str,
        rank: int,
        cfg: WorkloadConfig,
        device_idx: int = 0,
    ):
        self.cfg = cfg
        self.rank = rank
        self.label = device_idx

        logging.info(f"GPU {rank}: compiling kernel (first run may take 15-30s)...")
        mod = SourceModule(
            program_src,
            options=["--use_fast_math", "-O3", "--ptxas-options=-O3"],
            no_extern_c=True,
        )
        self.kern = mod.get_function("ed25519_scan")
        logging.info(f"GPU {rank}: kernel ready")

        self.block_dim = 256
        self.grid_dim = (cfg.batch_size + self.block_dim - 1) // self.block_dim
        cfg._stride = self.block_dim * self.grid_dim

        self._setup_buffers()
        self._interval_keys = 0
        self._lifetime_keys = 0
        self._last_report = time.monotonic()
        self._tick_count = 0

    def _setup_buffers(self) -> None:
        self.d_seed = cuda.mem_alloc(32)
        self.d_result = cuda.mem_alloc(33)
        self.d_sweep_len = cuda.mem_alloc(1)
        self.d_rank = cuda.mem_alloc(1)

        cuda.memcpy_htod(
            self.d_sweep_len,
            np.array([self.cfg.sweep_bytes], dtype=np.uint8),
        )
        cuda.memcpy_htod(
            self.d_rank,
            np.array([0], dtype=np.uint8),
        )
        self.result_host = np.zeros(33, dtype=np.uint8)

    def tick(self) -> np.ndarray:
        """Run one batch on the GPU and return the 33-byte result buffer."""
        cuda.memcpy_htod(self.d_seed, self.cfg.seed)

        self.kern(
            self.d_seed, self.d_result, self.d_sweep_len, self.d_rank,
            block=(self.block_dim, 1, 1),
            grid=(self.grid_dim, 1),
        )

        self.cfg.step()

        cuda.memcpy_dtoh(self.result_host, self.d_result)

        keys_this_tick = self.block_dim * self.grid_dim
        self._interval_keys += keys_this_tick
        self._lifetime_keys += keys_this_tick
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
    device_selection: Optional[List[int]] = None,
) -> None:
    """Long-lived process: mine on one GPU, push found keys to `hits` queue."""
    try:
        cuda.init()

        if device_selection is not None:
            dev_idx = device_selection[rank]
        else:
            all_gpus = discover_gpus()
            dev_idx = all_gpus[rank]

        device = cuda.Device(dev_idx)
        ctx = device.make_context()

        try:
            os.sched_setaffinity(0, {rank % (os.cpu_count() or 1)})
        except (AttributeError, OSError):
            pass

        miner = GPUMiner(
            program_src=cfg.program_src,
            rank=rank,
            cfg=cfg,
            device_idx=dev_idx,
        )

        while not halt.value:
            result = miner.tick()
            counters[rank] = miner._lifetime_keys

            if result[0]:
                hits.put(bytes(result[1:33]))

                miner.result_host[:] = 0
                cuda.memcpy_htod(miner.d_result, miner.result_host)
                miner.cfg.randomize()

        ctx.pop()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.exception(f"GPU {rank} crashed: {exc}")
