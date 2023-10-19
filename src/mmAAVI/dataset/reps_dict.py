import collections as ct
from typing import Any

from .data_grid import DataGrid


class Reps(ct.UserDict):

    def link(self, data):
        self._mmodata = data
        for k, repi in self.data.items():
            self.check(repi, k)
        return self

    def check(self, repi: DataGrid, key: str):
        assert isinstance(repi, DataGrid), \
            "the element of reps must be a DataGrid"
        assert repi.batch_dims_dict == self._mmodata.batch_dims_dict, \
            "the batch_dims_dict of rep must be equal to batch_dims_dict of X"
        assert repi.omics_names == self._mmodata.omics_names, \
            "the omics_names of rep must be equal to omics_names of X"

        for bk, ok, dati in self._mmodata.X.items():
            assert not ((dati is None) ^ (repi._grid[bk][ok] is None)), \
                "(%s, %s) of X is %s, but (%s, %s) of rep %s is %s" % (
                bk, ok, str(dati), bk, ok, key, str(repi._grid[bk][ok])
            )

    def __setitem__(self, key: str, item: Any) -> None:
        if not isinstance(item, DataGrid):
            item = DataGrid(item)
        if hasattr(self, "_mmodata"):
            self.check(item, key)
        return super().__setitem__(key, item)
