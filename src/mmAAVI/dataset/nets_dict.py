import collections as ct
from typing import Any

from .data_grid import DataGrid


class Nets(ct.UserDict):

    def link(self, data):
        self._mmodata = data
        for neti in self.data.values():
            self.check(neti)
            neti.expand(
                self._mmodata.omics_names, self._mmodata.omics_names,
                self._mmodata.omics_dims, self._mmodata.omics_dims
            )
        return self

    def check(self, neti: DataGrid):
        assert isinstance(neti, DataGrid), \
            "the element of nets must be a DataGrid"
        assert all(i in self._mmodata.omics_names for i in neti.batch_names), \
            "the batch_names of net must be element of omis_names of X"
        assert all(i in self._mmodata.omics_names for i in neti.omics_names), \
            "the omics_names of net must be element of omis_names of X"
        for i, j, dati in neti.items():
            if dati is not None:
                assert dati.shape[0] == self._mmodata.omics_dims_dict[i], (
                    ("the net shape[0] of %s and %s is %d, "
                     "which is not equal to ndim of %s (%d)") %
                    (i, j, dati.shape[0], i,
                     self._mmodata.omics_dims_dict[i])
                )
                assert dati.shape[1] == self._mmodata.omics_dims_dict[j], (
                    ("the net shape[1] of %s and %s is %d, "
                     "which is not equal to ndim of %s (%d)") %
                    (i, j, dati.shape[1], j,
                     self._mmodata.omics_dims_dict[j])
                )

    def __setitem__(self, key: str, item: Any) -> None:
        if not isinstance(item, DataGrid):
            item = DataGrid(item)
        if hasattr(self, "_mmodata"):
            self.check(item)
            item.expand(
                self._mmodata.omics_names, self._mmodata.omics_names,
                self._mmodata.omics_dims, self._mmodata.omics_dims
            )
        return super().__setitem__(key, item)

    def expand_full_names(self) -> None:
        """ 将所有的omics names补齐，用None """
        for neti in self.values():
            neti.expand(self._mmodata.omics_names, self._mmodata.omics_names,
                        self._mmodata.omics_dims, self._mmodata.omics_dims)
