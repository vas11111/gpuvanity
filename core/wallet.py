import json
from pathlib import Path
from typing import List, Optional, Tuple

from base58 import b58encode
from nacl.signing import SigningKey


def derive_address(secret: bytes) -> str:
    """Produce the Base58 public address from a 32-byte secret."""
    return b58encode(bytes(SigningKey(secret).verify_key)).decode()


def export_keypair(secret: bytes, directory: str) -> str:
    """Write a Solana-CLI-compatible keypair JSON. Returns the public address."""
    sk = SigningKey(secret)
    pk_bytes = bytes(sk.verify_key)
    address = b58encode(pk_bytes).decode()

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{address}.json").write_text(json.dumps(list(secret + pk_bytes)))

    return address


def identify_match(
    address: str,
    prefixes: List[str],
    suffixes: List[str],
    case_sensitive: bool,
    match_all: bool = False,
) -> Optional[Tuple[str, str]]:
    """Return (tag, pattern) for the first matching rule, or None.

    In OR mode (default): returns ("pfx", p) or ("sfx", s) for the first hit.
    In AND mode (match_all): returns ("both", "p+s") only if both a prefix
    and a suffix match simultaneously.
    """
    cmp = address if case_sensitive else address.lower()

    if match_all:
        matched_pfx = None
        for p in prefixes:
            t = p if case_sensitive else p.lower()
            if cmp.startswith(t):
                matched_pfx = p
                break
        if matched_pfx is None:
            return None

        for s in suffixes:
            t = s if case_sensitive else s.lower()
            if cmp.endswith(t):
                return "both", f"{matched_pfx}+{s}"
        return None

    for p in prefixes:
        t = p if case_sensitive else p.lower()
        if cmp.startswith(t):
            return "pfx", p

    for s in suffixes:
        t = s if case_sensitive else s.lower()
        if cmp.endswith(t):
            return "sfx", s

    return None


def match_targets(
    address: str,
    targets: List[dict],
    case_sensitive: bool,
) -> Optional[Tuple[str, str]]:
    """Check address against a list of targets with independent match modes.

    Each target dict has optional 'prefix' and 'suffix' keys.
    Both present → AND mode, one present → match that part only.
    Returns (tag, pattern) for the first matching target, or None.
    """
    cmp = address if case_sensitive else address.lower()

    for t in targets:
        prefix = t.get("prefix", "")
        suffix = t.get("suffix", "")

        if prefix:
            p = prefix if case_sensitive else prefix.lower()
            if not cmp.startswith(p):
                continue

        if suffix:
            s = suffix if case_sensitive else suffix.lower()
            if not cmp.endswith(s):
                continue

        if prefix and suffix:
            return "both", f"{prefix}+{suffix}"
        elif prefix:
            return "pfx", prefix
        else:
            return "sfx", suffix

    return None
