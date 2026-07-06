from __future__ import annotations
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, u: int, v: int) -> str:
        raise NotImplementedError
