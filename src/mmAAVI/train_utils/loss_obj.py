from typing import Mapping
from collections import UserDict

from torch import Tensor as T


class BatchLoss(UserDict):
    def __init__(self, **weights: dict[str, float]) -> None:
        super().__init__()
        self._weights = weights

    def __setitem__(self, key: str, item: T) -> None:
        if key in self.data:
            raise ValueError("duplicated keys")
        return super().__setitem__(key, item)

    def update(self, new: Mapping):
        for key in new:
            if key in self.data:
                raise ValueError("duplicated keys")
        super().update(new)

    @property
    def total(self) -> T:
        """The total property."""
        if hasattr(self, "_total"):
            return self._total
        self._total = self.metric
        for k, v in self.data.items():
            if k.startswith("disc"):
                self._total += self._weights[k] * v
        return self._total

    @property
    def metric(self):
        """The total property."""
        if hasattr(self, "_metric"):
            return self._metric
        self._metric = 0.0
        for k, v in self.data.items():
            if not k.startswith("disc"):
                self._metric += self._weights[k] * v
        return self._metric
