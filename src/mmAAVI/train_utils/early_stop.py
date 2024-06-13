import logging

import numpy as np
from torch.optim import Optimizer

from .history import History


class EarlyStopper:

    def __init__(
        self, metric: str, phase: str = "valid", patient: int = 10
    ) -> None:
        self._patient = patient
        self._metric = metric
        self._phase = phase

        self._mode = "min" if phase in ["train", "valid"] else "max"
        self._best_score = np.inf if self._mode == "min" else -np.inf
        self._cnt = 0

    def watch(self, history: History, ind: int = -1) -> None:
        # e = history._hist["score"]["epoch"][ind]
        score = history._hist[self._phase][self._metric][ind]

        if self._mode == "min":
            flag = score > self._best_score
        else:
            flag = score < self._best_score
        if flag:
            self._cnt += 1
        else:
            self._best_score = score
            self._cnt = 0

    def is_stop(self) -> bool:
        return self._cnt >= self._patient

    def print_msg(self):
        if self.is_stop():
            logging.info("stop because of no decreased loss")


class LearningRateUpdateEarlyStopper:

    def __init__(self, lr_init: float, max_update: int = 2):
        self._max_update = max_update
        self._previous_lr = lr_init
        self._lr_update_cnt = 0

    def watch(self, opt: Optimizer) -> None:
        current_lr = opt.param_groups[0]["lr"]
        if current_lr < self._previous_lr:
            self._lr_update_cnt += 1
            self._previous_lr = current_lr

    def is_stop(self) -> bool:
        return self._lr_update_cnt > self._max_update

    def print_msg(self):
        if self.is_stop():
            logging.info("stop because of too much lr update")
