from typing import Optional, Mapping, Dict
from collections import defaultdict

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class History:

    def __init__(self):
        self._hist = {}

    def append(
        self, epoch: int, record: Mapping[str, float],
        key: str = "train", log_writer: Optional[SummaryWriter] = None
    ):
        histi = self._hist.setdefault(key, defaultdict(list))
        histi["epoch"].append(epoch)
        for k, v in record.items():
            histi[k].append(v)
        if log_writer is not None:
            for k, v in record.items():
                log_writer.add_scalar("%s/%s" % (key, k), v, epoch)
            log_writer.flush()

    def to_dfs(self) -> Dict[str, pd.DataFrame]:
        return {k: pd.DataFrame.from_records(histi)
                for k, histi in self._hist.items()}

    def show_record(self, key: str, ind: int = -1) -> str:
        histi = self._hist[key]
        return "Epoch %s: %d, " % (key, histi["epoch"][ind]) + \
            ", ".join(["%s: %.2f" % (k, seq[ind]) for k, seq in histi.items()])
