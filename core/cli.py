import json
import logging
import multiprocessing
import os
import queue
import signal
import sys
import time
from ctypes import c_uint64
from multiprocessing import Array, Process, Queue, Value
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import click
import pycuda.driver as cuda

from core.devices import discover_gpus, get_device_info, pick_devices
from core.miner import _fmt_count, mine_loop
from core.program import assert_base58, build_program_source
from core.wallet import derive_address, export_keypair, identify_match, match_targets
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


def _jobs_done(tally: Dict[str, int], limits: Dict[str, int]) -> bool:
    return all(tally.get(k, 0) >= v for k, v in limits.items())


def _load_targets(path: str) -> List[dict]:
    with open(path) as f:
        targets = json.load(f)
    if not isinstance(targets, list) or not targets:
        raise click.ClickException("Config must be a non-empty JSON array of targets.")
    for i, t in enumerate(targets):
        pfx = t.get("prefix", "")
        sfx = t.get("suffix", "")
        if not pfx and not sfx:
            raise click.ClickException(f"Target {i}: must have at least 'prefix' or 'suffix'.")
        if pfx:
            assert_base58(f"target {i} prefix", pfx)
        if sfx:
            assert_base58(f"target {i} suffix", sfx)
    return targets


def _target_key(t: dict) -> str:
    pfx, sfx = t.get("prefix", ""), t.get("suffix", "")
    if pfx and sfx:
        return f"both_{pfx}+{sfx}"
    if pfx:
        return f"pfx_{pfx}"
    return f"sfx_{sfx}"


@click.command(context_settings={"show_default": True})
@click.option("--prefix", default="", help="Comma-separated prefix targets.")
@click.option("--suffix", default="", help="Comma-separated suffix targets.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="JSON config file with targets (overrides --prefix/--suffix/--match-all).")
@click.option("--count", default=1, type=int, help="Keys per target (0 = run forever).")
@click.option("--output-dir", default="./keys", type=click.Path(file_okay=False, dir_okay=True), help="Root output directory.")
@click.option("--select-device/--no-select-device", default=False, help="Interactive GPU picker.")
@click.option("--batch-exp", default=DEFAULT_BATCH_EXP, type=int, help="Batch size exponent (26-30 recommended).")
@click.option("--case-sensitive/--no-case-sensitive", default=True, help="Pattern matching mode.")
@click.option("--match-all", is_flag=True, default=False, help="Require BOTH prefix AND suffix to match (default: match ANY).")
@click.option("--devices", is_flag=True, help="Print GPUs and exit.")
def main(
    prefix: str,
    suffix: str,
    config_path: Optional[str],
    count: int,
    output_dir: str,
    select_device: bool,
    batch_exp: int,
    case_sensitive: bool,
    match_all: bool,
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

    targets: Optional[List[dict]] = None

    if config_path:
        targets = _load_targets(config_path)
        pfx_set: set = set()
        sfx_set: set = set()
        for t in targets:
            pfx, sfx = t.get("prefix", ""), t.get("suffix", "")
            if pfx and sfx:
                if len(pfx) >= len(sfx):
                    pfx_set.add(pfx)
                else:
                    sfx_set.add(sfx)
            elif pfx:
                pfx_set.add(pfx)
            elif sfx:
                sfx_set.add(sfx)
        pfx_list = list(pfx_set)
        sfx_list = list(sfx_set)
    else:
        pfx_list = _split_csv(prefix)
        sfx_list = _split_csv(suffix)

    if not pfx_list and not sfx_list:
        click.echo("Provide at least --prefix or --suffix (or use --config).")
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    if not config_path:
        if match_all and (not pfx_list or not sfx_list):
            click.echo("--match-all requires both --prefix and --suffix.")
            sys.exit(1)
        for p in pfx_list:
            assert_base58("prefix", p)
        for s in sfx_list:
            assert_base58("suffix", s)

    n_gpus, gpu_sel = _detect_gpus(select_device)
    _tune_process()

    tally: Dict[str, int] = {}
    limits: Dict[str, int] = {}

    if targets:
        for t in targets:
            key = _target_key(t)
            tally[key] = 0
            limits[key] = t.get("count", count)
    elif match_all:
        for p in pfx_list:
            for s in sfx_list:
                k = f"both_{p}+{s}"
                tally[k] = 0
                limits[k] = count
    else:
        for p in pfx_list:
            tally[f"pfx_{p}"] = 0
            limits[f"pfx_{p}"] = count
        for s in sfx_list:
            tally[f"sfx_{s}"] = 0
            limits[f"sfx_{s}"] = count

    forever = all(v == 0 for v in limits.values())

    logging.info(f"{n_gpus} GPU(s) | batch 2^{batch_exp} = {1 << batch_exp:,} keys/iter/GPU")
    if targets:
        for t in targets:
            pfx, sfx = t.get("prefix", ""), t.get("suffix", "")
            mode = "AND" if pfx and sfx else "prefix" if pfx else "suffix"
            c = t.get("count", count)
            logging.info(f"  Target: {_target_key(t)} ({mode}) x{c}")
    else:
        parts = []
        if pfx_list:
            parts.append(f"prefix=[{', '.join(pfx_list)}]")
        if sfx_list:
            parts.append(f"suffix=[{', '.join(sfx_list)}]")
        mode_label = "AND" if match_all else "OR"
        logging.info(f"Targets: {', '.join(parts)} | mode={mode_label}")
    logging.info(f"{'Continuous' if forever else 'Per-target counts'} | case_sensitive={case_sensitive}")

    sweep_bytes = (batch_exp + 7) >> 3
    src = build_program_source(
        tuple(pfx_list), tuple(sfx_list), case_sensitive, sweep_bytes,
        match_all=False if targets else match_all,
    )

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

    def _check_hit(secret: bytes) -> None:
        address = derive_address(secret)
        if targets:
            hit = match_targets(address, targets, case_sensitive)
        else:
            hit = identify_match(address, pfx_list, sfx_list, case_sensitive, match_all)
        if hit is None:
            return
        tag, pattern = hit
        key = f"{tag}_{pattern}"
        limit = limits.get(key, 0)
        if not forever and limit > 0 and tally.get(key, 0) >= limit:
            return
        folder = _dest(output_dir, tag, pattern)
        saved = export_keypair(secret, str(folder))
        tally[key] = tally.get(key, 0) + 1
        logging.info(f"FOUND {key}: {saved} ({tally[key]}/{limit})")
        if not forever and _jobs_done(tally, limits):
            logging.info("All targets satisfied")
            halt.value = 1

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

        _check_hit(secret)

    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()

    cumulative_searched = sum(counters)

    while not hits.empty():
        try:
            secret = hits.get_nowait()
            _check_hit(secret)
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
