import logging
import os
import queue
import signal
import sys
import time
from ctypes import c_uint64
from multiprocessing import Array, Process, Queue, Value
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import pycuda.driver as cuda

from core.devices import discover_gpus, get_device_info, pick_devices
from core.miner import _fmt_count, mine_loop
from core.program import assert_base58, build_program_source
from core.wallet import derive_address, export_keypair, identify_match
from core.workload import DEFAULT_BATCH_EXP, WorkloadConfig

logging.basicConfig(
    level="INFO",
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

_STATUS_INTERVAL = 10.0
_DRAIN_TIMEOUT = 0.25


def _split_csv(raw: str) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _dest(base: str, tag: str, pattern: str) -> Path:
    return Path(base) / f"{tag}_{pattern}"


def _tune_process() -> None:
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except (AttributeError, OSError):
        pass
    try:
        os.nice(-5)
    except (AttributeError, OSError, PermissionError):
        pass


def _detect_gpus(manual: bool) -> Tuple[int, Optional[List[int]]]:
    cuda.init()
    if manual:
        sel = pick_devices()
        return len(sel), sel
    n = len(discover_gpus())
    if n == 0:
        logging.error("No GPUs detected")
        sys.exit(1)
    return n, None


def _jobs_done(tally: Dict[str, int], target: int) -> bool:
    return all(v >= target for v in tally.values())


@click.command(context_settings={"show_default": True})
@click.option("--prefix", default="", help="Comma-separated prefix targets.")
@click.option("--suffix", default="", help="Comma-separated suffix targets.")
@click.option("--count", default=1, type=int, help="Keys per target (0 = run forever).")
@click.option("--output-dir", default="./keys", type=click.Path(file_okay=False, dir_okay=True), help="Root output directory.")
@click.option("--select-device/--no-select-device", default=False, help="Interactive GPU picker.")
@click.option("--batch-exp", default=DEFAULT_BATCH_EXP, type=int, help="Batch size exponent (26-30 recommended).")
@click.option("--case-sensitive/--no-case-sensitive", default=True, help="Pattern matching mode.")
@click.option("--devices", is_flag=True, help="Print GPUs and exit.")
def main(
    prefix: str,
    suffix: str,
    count: int,
    output_dir: str,
    select_device: bool,
    batch_exp: int,
    case_sensitive: bool,
    devices: bool,
):
    """Solana vanity address miner -- GPU accelerated via CUDA."""
    if devices:
        cuda.init()
        n = cuda.Device.count()
        for i in range(n):
            info = get_device_info(i)
            click.echo(
                f"  [{i}] {info['name']}  --  {info['sms']} SMs, {info['vram_gb']:.1f} GB"
            )
        return

    pfx_list = _split_csv(prefix)
    sfx_list = _split_csv(suffix)

    if not pfx_list and not sfx_list:
        click.echo("Provide at least --prefix or --suffix.")
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    for p in pfx_list:
        assert_base58("prefix", p)
    for s in sfx_list:
        assert_base58("suffix", s)

    n_gpus, gpu_sel = _detect_gpus(select_device)
    _tune_process()

    forever = count == 0

    tally: Dict[str, int] = {}
    for p in pfx_list:
        tally[f"pfx_{p}"] = 0
    for s in sfx_list:
        tally[f"sfx_{s}"] = 0

    logging.info(f"{n_gpus} GPU(s) | batch 2^{batch_exp} = {1 << batch_exp:,} keys/iter/GPU")
    parts = []
    if pfx_list:
        parts.append(f"prefix=[{', '.join(pfx_list)}]")
    if sfx_list:
        parts.append(f"suffix=[{', '.join(sfx_list)}]")
    logging.info(f"Targets: {', '.join(parts)} | {'continuous' if forever else f'{count} each'}")

    src = build_program_source(tuple(pfx_list), tuple(sfx_list), case_sensitive)

    halt = Value("i", 0)

    def _on_signal(sig, frame):
        if not halt.value:
            logging.info("Signal received, finishing current GPU batches...")
            halt.value = 1

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    t0 = time.monotonic()
    counters = Array(c_uint64, n_gpus, lock=False)
    hits: Queue = Queue()

    workers: List[Process] = []
    for i in range(n_gpus):
        w = Process(
            target=mine_loop,
            args=(i, WorkloadConfig(src, batch_exp), halt, hits, counters, gpu_sel),
            daemon=True,
        )
        w.start()
        workers.append(w)

    last_status = time.monotonic()
    prev_searched = 0

    while not halt.value:
        try:
            secret = hits.get(timeout=_DRAIN_TIMEOUT)
        except queue.Empty:
            now = time.monotonic()
            if now - last_status >= _STATUS_INTERVAL:
                dt = now - t0
                total_searched = sum(counters)
                delta = total_searched - prev_searched
                interval = now - last_status
                agg_rate = delta / (interval * 1e6) if interval > 0 else 0.0
                summary = ", ".join(f"{k}: {v}" for k, v in tally.items())
                logging.info(
                    f"{dt:.0f}s elapsed | "
                    f"{_fmt_count(total_searched)} searched | "
                    f"{agg_rate:.2f} MH/s aggregate | "
                    f"{sum(tally.values())} found [{summary}]"
                )
                prev_searched = total_searched
                last_status = now
            continue

        address = derive_address(secret)
        hit = identify_match(address, pfx_list, sfx_list, case_sensitive)

        if hit is None:
            continue

        tag, pattern = hit
        key = f"{tag}_{pattern}"

        if not forever and tally[key] >= count:
            continue

        folder = _dest(output_dir, tag, pattern)
        export_keypair(secret, str(folder))
        tally[key] += 1

        if not forever and _jobs_done(tally, count):
            logging.info("All targets satisfied")
            halt.value = 1

    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()

    cumulative_searched = sum(counters)

    while not hits.empty():
        try:
            secret = hits.get_nowait()
            address = derive_address(secret)
            hit = identify_match(address, pfx_list, sfx_list, case_sensitive)
            if hit:
                tag, pattern = hit
                key = f"{tag}_{pattern}"
                if forever or tally[key] < count:
                    folder = _dest(output_dir, tag, pattern)
                    export_keypair(secret, str(folder))
                    tally[key] += 1
        except queue.Empty:
            break

    total_time = time.monotonic() - t0
    total_keys = sum(tally.values())
    avg_rate = cumulative_searched / (total_time * 1e6) if total_time > 0 else 0.0
    logging.info(
        f"Done: {total_keys} keys found in {total_time:.1f}s | "
        f"{_fmt_count(cumulative_searched)} addresses searched | "
        f"avg {avg_rate:.2f} MH/s"
    )
    for k, v in tally.items():
        if v:
            logging.info(f"  {k}: {v}")
