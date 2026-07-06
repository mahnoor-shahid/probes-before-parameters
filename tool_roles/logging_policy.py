from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LoggingPolicy:
    """
    Conservative logging policy pi0:
      execute iff u==v (default), else reject.
    Can also be parameterized by a predicate.
    """
    predicate_execute: Callable[[int, int], bool]

    def act(self, u: int, v: int) -> str:
        return "execute" if self.predicate_execute(u, v) else "reject"


def diagonal_logging_policy() -> LoggingPolicy:
    return LoggingPolicy(predicate_execute=lambda u, v: u == v)
