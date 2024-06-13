from copy import deepcopy

import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .history import History


class Checkpointer:
    def __init__(self, metric: str, phase: str = "valid") -> None:
        # self._metric = metric

        self._k1, self._k2 = phase, metric
        self._mode = "min" if self._k1 in ["train", "valid"] else "max"

        self._best = {
            "metric": metric,
            "epoch": -1,
            "value": -np.inf if self._mode == "max" else np.inf,
        }
        self._model_state_dict = None
        self._flag_reuse = False

    def watch(
        self, history: History, ind: int = -1
    ) -> float:  # return the monitored score
        self._flag_reuse = False
        self._scorei = history._hist[self._k1][self._k2][ind]
        self._epochi = history._hist[self._k1]["epoch"][ind]
        return self._scorei

    def update_best(self, model: nn.Module, verbose: int = 0) -> bool:
        if self._flag_reuse:
            raise ValueError(
                "the score stoared in early stopper was used twice."
            )
        self._flag_reuse = True
        if self._mode == "max":
            flag = self._scorei > self._best["value"]
        else:
            flag = self._scorei < self._best["value"]
        if flag:
            if verbose >= 3:
                tqdm.write(
                    "Best model at %d, %s from %.4f to %.4f"
                    % (
                        self._epochi,
                        self._k2,
                        self._best["value"],
                        self._scorei,
                    )
                )
            self._best["epoch"] = self._epochi
            self._best["value"] = self._scorei
            self._model_state_dict = deepcopy(model.state_dict())
            return True
        return False

    def apply_state_dict(self, model: nn.Module) -> None:
        assert self._model_state_dict is not None
        model.load_state_dict(self._model_state_dict)
