from typing import Dict, Tuple
from collections import defaultdict

from torch import Tensor


class LossAccumulator:

    def __init__(self) -> None:
        self.init()

    def init(self) -> None:
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, **losses: Dict[str, Tuple[Tensor, int]]) -> None:
        for k, v in losses.items():
            self.total[k] += v if isinstance(v, float) else v.item()
            self.count[k] += 1

    def calc(self) -> Dict[str, float]:
        return {k: v / self.count[k] for k, v in self.total.items()}
