from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable
