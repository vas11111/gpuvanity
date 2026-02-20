import logging
from pathlib import Path
from typing import Tuple

from base58 import b58decode

_CUDA_SOURCE = Path(__file__).parent / "cuda" / "kernel.cu"


def assert_base58(label: str, text: str) -> None:
    """Abort if text contains characters outside the Base58 alphabet."""
    if not text:
        return
    try:
        b58decode(text)
    except ValueError as exc:
        logging.error(f"Bad Base58 in {label}: {exc}")
        raise SystemExit(1)


def build_program_source(
    prefixes: Tuple[str, ...],
    suffixes: Tuple[str, ...],
    case_sensitive: bool,
) -> str:
    """Read the CUDA template and inject match parameters for all patterns."""
    pfx_encoded = [list(p.encode()) for p in prefixes] if prefixes else []
    sfx_encoded = [list(s.encode()) for s in suffixes] if suffixes else []

    n_pfx = len(pfx_encoded)
    longest_pfx = max((len(e) for e in pfx_encoded), default=0)
    for e in pfx_encoded:
        e += [0] * (longest_pfx - len(e))

    n_sfx = len(sfx_encoded)
    longest_sfx = max((len(e) for e in sfx_encoded), default=0)
    for e in sfx_encoded:
        e += [0] * (longest_sfx - len(e))

    if not _CUDA_SOURCE.exists():
        raise FileNotFoundError(f"Missing CUDA source: {_CUDA_SOURCE}")

    raw = _CUDA_SOURCE.read_text().splitlines(keepends=True)

    begin_idx = None
    end_idx = None
    for idx, ln in enumerate(raw):
        if "BEGIN INJECTED PARAMETERS" in ln:
            begin_idx = idx
        elif "END INJECTED PARAMETERS" in ln:
            end_idx = idx
            break

    if begin_idx is None or end_idx is None:
        raise RuntimeError("Could not locate injection markers in CUDA source")

    block = [raw[begin_idx]]
    block.append(f"#define N {n_pfx}\n")
    block.append(f"#define L {max(longest_pfx, 1) if n_pfx == 0 else longest_pfx}\n")
    if n_pfx > 0:
        cells = ", ".join(
            "{" + ", ".join(map(str, row)) + "}" for row in pfx_encoded
        )
        block.append(
            f"__constant__ unsigned char PREFIXES[{n_pfx}][{longest_pfx}] = {{{cells}}};\n"
        )
    block.append(f"#define NS {n_sfx}\n")
    block.append(f"#define SL {max(longest_sfx, 1) if n_sfx == 0 else longest_sfx}\n")
    if n_sfx > 0:
        cells = ", ".join(
            "{" + ", ".join(map(str, row)) + "}" for row in sfx_encoded
        )
        block.append(
            f"__constant__ unsigned char SUFFIXES[{n_sfx}][{longest_sfx}] = {{{cells}}};\n"
        )
    block.append(
        f"__constant__ bool CASE_SENSITIVE = {str(case_sensitive).lower()};\n"
    )
    block.append(raw[end_idx])

    raw[begin_idx : end_idx + 1] = block
    return "".join(raw)
