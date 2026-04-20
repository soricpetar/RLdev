from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class Space:
    def sample(self):
        raise NotImplementedError


@dataclass
class Box(Space):
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: type | np.dtype = np.float32

    def sample(self):
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


@dataclass
class Discrete(Space):
    n: int

    def sample(self):
        return int(np.random.randint(0, self.n))


@dataclass
class Dict(Space):
    spaces: dict[str, Space]

    def sample(self):
        return {k: v.sample() for k, v in self.spaces.items()}
