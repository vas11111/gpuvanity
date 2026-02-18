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
) -> Optional[Tuple[str, str]]:
    """Return ("pfx"|"sfx", pattern) for the first matching rule, or None."""
    cmp = address if case_sensitive else address.lower()

    for p in prefixes:
        t = p if case_sensitive else p.lower()
        if cmp.startswith(t):
            return "pfx", p

    for s in suffixes:
        t = s if case_sensitive else s.lower()
        if cmp.endswith(t):
            return "sfx", s

    return None
