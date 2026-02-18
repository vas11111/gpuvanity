import secrets

import numpy as np

DEFAULT_BATCH_EXP = 28
DEFAULT_WORKGROUP = 256


class WorkloadConfig:
    """Describes one unit of GPU work: seed, batch dimensions, compiled source."""

    __slots__ = (
        "batch_exp", "sweep_bytes", "batch_size",
        "workgroup_size", "program_src", "seed", "_stride",
    )

    def __init__(self, program_src: str, batch_exp: int):
        if not 0 <= batch_exp <= 255:
            raise ValueError(f"batch_exp out of range: {batch_exp}")

        self.batch_exp = batch_exp
        self.sweep_bytes = (batch_exp + 7) >> 3
        self.batch_size = 1 << batch_exp
        self.workgroup_size = DEFAULT_WORKGROUP
        self.program_src = program_src
        self._stride = int(np.uint64(1 << batch_exp))
        self.seed = self._new_seed()

    def _new_seed(self) -> np.ndarray:
        raw = np.frombuffer(secrets.token_bytes(32), dtype=np.uint8).copy()
        if self.sweep_bytes > 0:
            raw[-self.sweep_bytes:] = 0
        return raw

    def randomize(self) -> None:
        self.seed = self._new_seed()

    def step(self) -> None:
        n = int.from_bytes(self.seed.tobytes(), "big") + self._stride
        np.copyto(self.seed, np.frombuffer(n.to_bytes(32, "big"), dtype=np.uint8))
