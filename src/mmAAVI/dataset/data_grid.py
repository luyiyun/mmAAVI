import collections as ct
import collections.abc as cta
import inspect
import warnings
from typing import (Callable, Generator, Mapping, Optional, OrderedDict,
                    Sequence, Tuple, Union)

import numpy as np
from h5py import Dataset, Group
from scipy import sparse as sp

from .. import typehint as typ
from .utils import (convert_sparse_index, handle_sparse_slice, is_slice_none,
                    read_sparse_in_h5, save_sparse_in_h5)


def check_names(batch_names, omics_names):
    assert (
        batch_names is not None and omics_names is not None
    ), "please give batch_names and omics_names"
    assert (
        len(batch_names) > 0 or len(omics_names) > 0
    ), "batch_names and omics_names must be sequence with length > 1"
    assert all(
        [isinstance(i, str) for i in batch_names]
        + [isinstance(i, str) for i in omics_names]
    ), "batch_names and omics_names must be sequence of strings"


def _check_same_name_for_all_batches(data: typ.DATA_MAPPING):
    first = next(iter(data.values())).keys()
    if not all(v.keys() == first for v in data.values()):
        raise ValueError("Mapping[Mapping] data must has the same omics names")


def _check_same_len_names_dims(
    batch_names, omics_names, batch_dims, omics_dims
):
    if batch_dims is not None:
        assert len(batch_names) == len(
            batch_dims
        ), "the length of batch_names and batch_dims must be equal"
    if omics_dims is not None:
        assert len(omics_names) == len(
            omics_dims
        ), "the length of omics_names and omics_dims must be equal"


def _check_same_dim_across_axis(
    grid: typ.DATA_MAPPING,
    batch_dims: Optional[Sequence[int]] = None,
    omics_dims: Optional[Sequence[int]] = None,
):
    if batch_dims is None:
        batch_dims = [None] * len(grid)
    else:
        batch_dims = [("global", bdi) for bdi in batch_dims]
    if omics_dims is None:
        omics_dims = [None] * len(next(iter(grid.values())))
    else:
        omics_dims = [("global", odi) for odi in omics_dims]
    for i, (bk, dats) in enumerate(grid.items()):
        for j, (ok, dati) in enumerate(dats.items()):
            if dati is None:
                continue
            nri, nci = dati.shape
            if batch_dims[i] is None:
                batch_dims[i] = (ok, nri)
            else:
                assert (
                    nri == batch_dims[i][1]
                ), "batch dims is not equal between %s(%d) and %s(%d)" % (
                    *batch_dims[i],
                    ok,
                    nri,
                )
            if omics_dims[j] is None:
                omics_dims[j] = (bk, nci)
            else:
                assert (
                    nci == omics_dims[j][1]
                ), "omics dims is not equal between %s(%d) and %s(%d)" % (
                    *omics_dims[j],
                    bk,
                    nci,
                )


class DataGrid:
    """
                    shape[1]
                ┌──────────────┐
                │              │
             ┌─ ┌───┐ ┌───┐ ┌──┐
             │  └───┘ └───┘ └──┘
    shape[0] │  ┌───┐ ┌───┐ ┌──┐
             │  └───┘ └───┘ └──┘
             │  ┌───┐ ┌───┐ ┌──┐
             └─ └───┘ └───┘ └──┘
    """

    @classmethod
    def _check_data_format(cls, data: typ.DATA_GRID_INIT) -> str:
        def _check_block(data):
            return (
                isinstance(data, np.ndarray)
                or sp.issparse(data)
                or data is None
                or (
                    isinstance(data, tuple)
                    and len(data) == 2
                    and isinstance(data[0], int)
                    and isinstance(data[1], int)
                )
            )

        if isinstance(data, ct.OrderedDict) and all(
            isinstance(v, ct.OrderedDict) for v in data.values()
        ):
            check_res = []
            for vs in data.values():
                for v in vs.values():
                    check_res.append(_check_block(v))
            if all(check_res):
                return "odict[odict]"

        if isinstance(data, cta.Mapping) and all(
            isinstance(v, cta.Mapping) for v in data.values()
        ):
            check_res = []
            for vs in data.values():
                for v in vs.values():
                    check_res.append(_check_block(v))
            if all(check_res):
                return "map[map]"

        if isinstance(data, cta.Sequence):
            check_res_record, check_res_list = [], []
            for seqi in data:
                check_res_record.append(
                    isinstance(seqi, tuple)
                    and isinstance(seqi[0], str)
                    and isinstance(seqi[1], str)
                    and _check_block(seqi[2])
                )
                check_res_list.append(_check_block(seqi))
            if all(check_res_record):
                return "seq[(str, str, arr)]"
            if all(check_res_list):
                return "seq[arr]"

        raise ValueError(
            "the data must be one of odict[odict[arr]], "
            "mapping[mapping[arr]], "
            "Seq[arr], or Seq[(str, str, arr)], "
            "and arr must be one of ndarray,"
            " sparse array, None, or 2-tuple shape"
        )

    def _init_from_data(
        self,
        data: typ.DATA_GRID_INIT,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
        batch_dims: Optional[Sequence[int]] = None,
        omics_dims: Optional[Sequence[int]] = None,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        data_format = self._check_data_format(data)

        if data_format == "map[map]":
            if batch_names is None:
                warnings.warn("data is a mapping without batch order")
            if omics_names is None:
                warnings.warn("data is a mapping without omics order")

        if data_format in ["odict[odict]", "map[map]"]:
            # mapping[str, mapping[str, arr]]
            _check_same_name_for_all_batches(data)
            batch_names, omics_names = self._init_from_data_map(
                data, batch_names, omics_names
            )
        elif data_format == "seq[(str, str, arr)]":
            # [(b1, o1, dat1), (b2, o2, dat2), ...]
            batch_names, omics_names = self._init_from_data_record(
                data, batch_names, omics_names
            )
        elif data_format == "seq[arr]":
            # [dat1, dat2, dat3, ...]
            if batch_names is None and batch_dims is None:
                raise ValueError(
                    "please given batch information for Seq[arr] data"
                )
            if omics_names is None and omics_dims is None:
                raise ValueError(
                    "please given omics information for Seq[arr] data"
                )
            batch_names, omics_names = self._init_from_data_list(
                data, batch_names, omics_names, batch_dims, omics_dims
            )

        return batch_names, omics_names

    def _init_from_data_map(
        self,
        data: typ.DATA_MAPPING,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        if batch_names is None:
            batch_names = data.keys()
        if omics_names is None:
            omics_names = next(iter(data.values())).keys()

        self._grid = ct.OrderedDict()
        for bi in batch_names:
            if bi not in data.keys():
                grid_i = ct.OrderedDict((oi, None) for oi in omics_names)
            else:
                grid_i = ct.OrderedDict()
                for oi in omics_names:
                    datai = data[bi].get(oi, None)
                    if isinstance(datai, tuple) and len(datai) == 2:
                        datai = sp.csr_array(datai)
                    grid_i[oi] = datai
            self._grid[bi] = grid_i

        return batch_names, omics_names

    def _init_from_data_record(
        self,
        data: typ.DATA_RECORD,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        if batch_names is None:
            batch_names = []
            for bi, _, _ in data:
                if bi not in batch_names:
                    batch_names.append(bi)
        if omics_names is None:
            omics_names = []
            for _, oi, _ in data:
                if oi not in omics_names:
                    omics_names.append(oi)

        self._grid = ct.OrderedDict(
            (bi, ct.OrderedDict((oi, None) for oi in omics_names))
            for bi in batch_names
        )
        for bi, oi, dati in data:
            if isinstance(dati, tuple) and len(dati) == 2:
                dati = sp.csr_array(dati)
            self._grid[bi][oi] = dati

        return batch_names, omics_names

    def _init_from_data_list(
        self,
        data: typ.DATA_RECORD,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
        batch_dims: Optional[Sequence[int]] = None,
        omics_dims: Optional[Sequence[int]] = None,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        if batch_names is None:
            batch_names = ["batch%d" % (i + 1) for i in range(len(batch_dims))]
        if omics_names is None:
            omics_names = ["omics%d" % (i + 1) for i in range(len(omics_dims))]

        seq_len, i = len(data), 0
        self._grid = ct.OrderedDict()
        for bi in batch_names:
            grid_i = ct.OrderedDict()
            for oi in omics_names:
                if i >= seq_len:
                    grid_i[oi] = None
                else:
                    datai = data[i]
                    if isinstance(datai, tuple) and len(datai) == 2:
                        datai = sp.csr_array(datai)
                    grid_i[oi] = datai
                i += 1
            self._grid[bi] = grid_i

        return batch_names, omics_names

    def _init_from_names(
        self, batch_names: Sequence[str], omics_names: Sequence[str]
    ) -> None:
        self._grid = ct.OrderedDict()
        for bi in batch_names:
            self._grid[bi] = ct.OrderedDict((oi, None) for oi in omics_names)

    def _set_names_dims(
        self,
        batch_names: Sequence[str],
        omics_names: Sequence[str],
        batch_dims: Optional[Sequence[int]] = None,
        omics_dims: Optional[Sequence[int]] = None,
    ) -> None:
        # 检查names的维度和dims是否相同
        _check_same_len_names_dims(
            batch_names, omics_names, batch_dims, omics_dims
        )
        _check_same_dim_across_axis(self._grid, batch_dims, omics_dims)

        # one_batch = next(iter(self._grid.values()))
        self._nbatch = len(batch_names)
        self._nomics = len(omics_names)
        self._batch_names = list(batch_names)
        self._omics_names = list(omics_names)

        if batch_dims is None:
            self._batch_dims = []
            for dats in self._grid.values():
                for dati in dats.values():
                    if dati is not None:
                        self._batch_dims.append(dati.shape[0])
                        break
                else:
                    self._batch_dims.append(0)
        else:
            self._batch_dims = list(batch_dims)
        if omics_dims is None:
            self._omics_dims = []
            for ok in self._omics_names:
                for bk in self._batch_names:
                    dati = self._grid[bk][ok]
                    if dati is not None:
                        self._omics_dims.append(dati.shape[1])
                        break
                else:
                    self._omics_dims.append(0)
        else:
            self._omics_dims = list(omics_dims)

        self._batch_dims_dict = ct.OrderedDict(
            zip(self.batch_names, self.batch_dims)
        )
        self._omics_dims_dict = ct.OrderedDict(
            zip(self.omics_names, self.omics_dims)
        )
        self._nobs = sum(self._batch_dims)
        self._nvar = sum(self._omics_dims)

        # 用于imp_miss的getitem
        self._batch_nonone = {}
        for ok in self._omics_names:
            self._batch_nonone[ok] = [
                i
                for i, dats in enumerate(self._grid.values())
                if dats[ok] is not None
            ]

    def set_null_for_zero_block(self, zero_dim: bool = False) -> "DataGrid":
        for bi in self._batch_names:
            for oi in self._omics_names:
                dati = self._grid[bi][oi]
                if dati is None:
                    continue
                if (isinstance(dati, np.ndarray) and (dati == 0).all()) or (
                    sp.issparse(dati) and dati.nnz == 0
                ):
                    self._grid[bi][oi] = None

        if zero_dim:
            # 检查所有为None的batch或omics，将其对应的dim设为0
            for i, bi in enumerate(self._batch_names):
                if all(v is None for v in self._grid[bi].values()):
                    self._batch_dims[i] = 0
                    self._batch_dims_dict[bi] = 0
            for i, oi in enumerate(self._omics_names):
                if all(self._grid[bk][oi] is None for bk in self._batch_names):
                    self._omics_dims[i] = 0
                    self._omics_dims_dict[oi] = 0

            self._nobs = sum(self._batch_dims)
            self._nvar = sum(self._omics_dims)

            self._batch_nonone = {}
            for ok in self._omics_names:
                self._batch_nonone[ok] = [
                    i
                    for i, dats in enumerate(self._grid.values())
                    if dats[ok] is not None
                ]

        return self

    def __init__(
        self,
        data: typ.DATA_GRID_INIT = None,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
        batch_dims: Optional[Sequence[int]] = None,
        omics_dims: Optional[Sequence[int]] = None,
    ) -> None:
        """
        batch/omics_dims：用来指定每个block的维度的，如果是-1，
            则表示根据data进行推断
        """
        if data is not None:
            batch_names, omics_names = self._init_from_data(
                data, batch_names, omics_names, batch_dims, omics_dims
            )
        else:
            if batch_names is None:
                if batch_dims is not None:
                    batch_names = [
                        "batch%d" % (i + 1) for i in range(len(batch_dims))
                    ]
                else:
                    batch_names = ["batch1"]
            if omics_names is None:
                if omics_dims is not None:
                    omics_names = [
                        "omics%d" % (i + 1) for i in range(len(omics_dims))
                    ]
                else:
                    omics_names = ["omics1"]
            self._init_from_names(batch_names, omics_names)

        self._set_names_dims(batch_names, omics_names, batch_dims, omics_dims)
        # self.set_null_for_zero_block()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._nobs, self._nvar)

    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def nvar(self) -> int:
        return self._nvar

    @property
    def nbatch(self) -> int:
        return self._nbatch

    @property
    def nomics(self) -> int:
        return self._nomics

    @property
    def batch_dims(self) -> Sequence[int]:
        return self._batch_dims

    @property
    def omics_dims(self) -> Sequence[int]:
        return self._omics_dims

    @property
    def batch_names(self) -> Sequence[str]:
        return self._batch_names

    @batch_names.setter
    def batch_names(self, value):
        assert (
            len(value) == self._nbatch
        ), "length of new batch_names is not correct."
        self._batch_names = list(value)

        new_grid = ct.OrderedDict()
        for nbk, dats in zip(value, self._grid.values()):
            new_grid[nbk] = dats
        self._grid = new_grid

        self._batch_dims_dict = ct.OrderedDict(
            (nbk, vi) for nbk, vi in zip(value, self._batch_dims)
        )

    @property
    def omics_names(self) -> Sequence[str]:
        return self._omics_names

    @omics_names.setter
    def omics_names(self, value):
        assert (
            len(value) == self._nomics
        ), "length of new omics_names is not correct."
        self._omics_names = list(value)

        new_grid = ct.OrderedDict()
        for bk, dats in self._grid.items():
            new_grid_b = ct.OrderedDict()
            for nok, dati in zip(value, dats.values()):
                new_grid_b[nok] = dati
            new_grid[bk] = new_grid_b
        self._grid = new_grid

        self._omics_dims_dict = ct.OrderedDict(
            (nok, vi) for nok, vi in zip(value, self._omics_dims)
        )

    @property
    def batch_dims_dict(self) -> OrderedDict[str, int]:
        return self._batch_dims_dict

    @property
    def omics_dims_dict(self) -> OrderedDict[str, int]:
        return self._omics_dims_dict

    def items(self) -> Generator[Tuple[str, str, typ.DATA_ELEM], None, None]:
        for bk, dats in self._grid.items():
            for ok, dati in dats.items():
                yield bk, ok, dati

    def __str__(self) -> str:
        n_char_index = max(len(i) for i in self._batch_names)
        n_char_nobs = max(
            1 if v == 0 else np.ceil(np.log10(v) + 1) for v in self._batch_dims
        )
        pattern = (
            "{:^%d} {:^%d} " % (n_char_index, n_char_nobs)
            + " ".join(["{:^10}"] * len(self._omics_names))
            + " \n"
        )
        msg = pattern.format("", "", *self._omics_names)
        msg += pattern.format("", "", *["(%d)" % v for v in self._omics_dims])
        for bk, dats in self._grid.items():
            msg += pattern.format(
                str(bk),
                "(%d)" % self._batch_dims_dict[bk],
                *[
                    "NULL" if v is None else v.__class__.__name__
                    for v in dats.values()
                ]
            )
        return msg

    def __getitem__(self, ind: typ.GRID_INDEX_ONE):
        """
        至多有4个坐标：batch_ind, omics_ind, row_ind, col_ind
        前两个坐标可以是：int、str、seq[int]、seq[str]、slice[int]
        后两个坐标可以是：int、seq[int]、slice[int]
          > 以上这个seq不能是tuple
        """

        if isinstance(ind, tuple) and len(ind) > 4:
            raise ValueError("length of index must be <= 4")

        if not isinstance(ind, tuple):
            bi, oi = ind, slice(None)
        else:
            bi, oi = ind[:2]

        if is_slice_none(bi) and is_slice_none(oi):
            # 如果[:, :]或[:, :, a, b]，直接跳过前两个index的索引过程
            flag_return_self = True
        else:
            flag_return_self = False

            grid_ind = []
            for ii, names in zip(
                [bi, oi], [self._batch_names, self._omics_names]
            ):
                if isinstance(ii, str):
                    grid_ind.append(ii)
                elif isinstance(ii, int):
                    grid_ind.append(names[ii])
                elif isinstance(ii, slice):
                    grid_ind.append(names[ii])
                elif isinstance(ii, cta.Sequence):
                    grid_ind.append(
                        [
                            names[iii] if isinstance(iii, int) else iii
                            for iii in ii
                        ]
                    )
            bi, oi = grid_ind

            # 如果bi和oi都是str，则表示只取其中的一个矩阵，不然，则返回一个新的datagrid
            if isinstance(bi, str) and isinstance(oi, str):
                subgrid = self._grid[bi][oi]
                flag_return_grid = False
            else:
                if isinstance(bi, str):
                    bi = [bi]
                if isinstance(oi, str):
                    oi = [oi]

                subgrid = ct.OrderedDict(
                    (
                        bk,
                        ct.OrderedDict(
                            (ok, dati) for ok, dati in dats.items() if ok in oi
                        ),
                    )
                    for bk, dats in self._grid.items()
                    if bk in bi
                )
                flag_return_grid = True

        # 如果只有前两个索引，或者两个索引是slice(None)，则直接返回datagrid
        if (
            not isinstance(ind, tuple)
            or len(ind) == 2
            or (len(ind) > 2 and all(is_slice_none(i) for i in ind[2:]))
        ):
            if flag_return_self:
                return self
            if flag_return_grid:
                return DataGrid(subgrid)
            else:
                return subgrid

        # 得到后两个索引
        ri = ind[2]
        ci = slice(None) if len(ind) != 4 else ind[3]

        # 如果前两个索引得到的array，则后两个索引等价于array[ri, ci]
        if not flag_return_self and not flag_return_grid:
            if subgrid is None:
                return None

            if sp.issparse(subgrid):
                ri, ci = convert_sparse_index(ri), convert_sparse_index(ci)
                # 当返回的是一行（或一列）数据时，将sparse格式更改为dense
                return handle_sparse_slice(subgrid[ri, ci])
            else:
                return subgrid[ri, ci]

        # 如果前两个索引得到的是datagrid，则是将整个datagrid看作一个大的
        #   array进行索引
        # TODO: 这种情况暂时没有实现
        raise NotImplementedError

    def select_rows_by_batch(
        self, indexes: Mapping[str, Sequence[int]], squeeze: bool = False
    ) -> "DataGrid":
        """
        根据每个batch提供的index来选择数据组成新的DataGrid，适合用于
        train test split
        """
        new_grid = ct.OrderedDict()
        for bk, bv in self._grid.items():
            if bk not in indexes:
                continue
            ind = indexes[bk]
            b_arrs = ct.OrderedDict()
            for ok, arr in bv.items():
                if arr is None:
                    b_arrs[ok] = None
                    continue
                b_arrs[ok] = arr[ind, :]
            new_grid[bk] = b_arrs
        if squeeze:
            return DataGrid(new_grid)
        return DataGrid(new_grid, omics_dims=self.omics_dims)

    def as_sparse_type(
        self, stype: str = "csr", only_sparse: bool = True
    ) -> "DataGrid":
        for bk in self._batch_names:
            for ok in self._omics_names:
                datai = self._grid[bk][ok]
                if datai is None:
                    continue
                if sp.issparse(datai):
                    if stype == "csr":
                        self._grid[bk][ok] = datai.tocsr()
                    elif stype == "coo":
                        self._grid[bk][ok] = datai.tocoo()
                    elif stype == "csc":
                        self._grid[bk][ok] = datai.tocsc()
                    else:
                        raise NotImplementedError
                elif not only_sparse and isinstance(datai, np.ndarray):
                    self._grid[bk][ok] = {
                        "csr": sp.csr_array,
                        "coo": sp.coo_array,
                        "csc": sp.csc_array,
                    }[stype](datai)
        return self

    def as_sparse_matrix(self) -> "DataGrid":
        """
        convert all sparse array to sparse matrix (better liqudity with R)
        """
        for bk in self._batch_names:
            for ok in self._omics_names:
                datai = self._grid[bk][ok]
                if datai is None:
                    continue

                if sp.issparse(datai):
                    cls_name = datai.__class__.__name__
                    if cls_name.endswith("array"):
                        self._grid[bk][ok] = {
                            "csr": sp.csr_matrix,
                            "coo": sp.coo_matrix,
                            "csc": sp.csc_matrix,
                        }[cls_name[:3]](datai)
        return self

    def as_sparse_array(self) -> "DataGrid":
        """convert all sparse matrix to sparse array"""
        for bk in self._batch_names:
            for ok in self._omics_names:
                datai = self._grid[bk][ok]
                if datai is None:
                    continue

                if sp.issparse(datai):
                    cls_name = datai.__class__.__name__
                    if cls_name.endswith("matrix"):
                        self._grid[bk][ok] = {
                            "csr": sp.csr_array,
                            "coo": sp.coo_array,
                            "csc": sp.csc_array,
                        }[cls_name[:3]](datai)
        return self

    def to_array(self, sparse: bool = True) -> typ.DATA_ELEM:
        """sparse: 是否输出稀疏矩阵格式"""
        res = []
        for bk, dats in self._grid.items():
            line = []
            for ok, dati in dats.items():
                if dati is None:
                    if sparse:
                        dati = sp.csr_array(
                            (
                                self.batch_dims_dict[bk],
                                self.omics_dims_dict[ok],
                            )
                        )
                    else:
                        dati = np.zeros(
                            (
                                self.batch_dims_dict[bk],
                                self.omics_dims_dict[ok],
                            )
                        )
                else:
                    if sp.issparse(dati) and not sparse:
                        dati = dati.todense()
                    elif isinstance(dati, np.ndarray) and sparse:
                        dati = sp.csr_array(dati)
                line.append(dati)
            line = sp.hstack(line) if sparse else np.concatenate(line, axis=1)
            res.append(line)
        return sp.vstack(res) if sparse else np.concatenate(res, axis=0)

    def expand(
        self,
        batch_names: Optional[Sequence[str]] = None,
        omics_names: Optional[Sequence[str]] = None,
        batch_dims: Optional[Sequence[str]] = None,
        omics_dims: Optional[Sequence[str]] = None,
    ) -> "DataGrid":
        if batch_names is None and omics_names is None:
            return self

        batch_names = (
            self._batch_names if batch_names is None else list(batch_names)
        )
        omics_names = (
            self._omics_names if omics_names is None else list(omics_names)
        )

        if (
            batch_names == self._batch_names
            and omics_names == self._omics_names
        ):
            return self

        new_grid = ct.OrderedDict()
        for bk in batch_names:
            gridi = ct.OrderedDict()
            for ok in omics_names:
                if bk in self._batch_names and ok in self._omics_names:
                    gridi[ok] = self._grid[bk][ok]
                else:
                    gridi[ok] = None
            new_grid[bk] = gridi
        self._grid = new_grid
        self._set_names_dims(batch_names, omics_names, batch_dims, omics_dims)
        return self

    def save_in_h5group(self, g: Group) -> None:
        g.attrs["batch_names"] = self._batch_names
        g.attrs["omics_names"] = self._omics_names
        g.attrs["batch_dims"] = self._batch_dims
        g.attrs["omics_dims"] = self._omics_dims
        # 有一些字符串名称不符合h5py的规则（比如以数字开头，这里在前面加上
        #   前缀来规避这个问题）
        ind_str = ["b_" + str(i) for i in self._batch_names]
        col_str = ["o_" + str(i) for i in self._omics_names]
        g.attrs["batch_names_"] = ind_str
        g.attrs["omics_names_"] = col_str
        for ri, ri_s in zip(self._batch_names, ind_str):
            g_ri = g.create_group(ri_s)
            for ci, ci_s in zip(self._omics_names, col_str):
                datai = self._grid[ri][ci]
                # nr, nc = self._batch_dims_dict[ri], self._omics_dims_dict[ci]
                if datai is None:
                    # if nr != 0 or nc != 0:
                    #     g_ri.create_dataset(ci_s, data=[nr, nc])
                    # else:
                    continue
                elif isinstance(datai, np.ndarray):
                    g_ri.create_dataset(ci_s, data=datai)
                elif sp.issparse(datai):
                    datai = datai.tocsr()
                    save_sparse_in_h5(g_ri, ci_s, datai)
                else:
                    raise ValueError(
                        "data element must be " "ndarray or scipy sparse array"
                    )

    @classmethod
    def load_from_h5group(self, g: Group) -> "DataGrid":
        index = g.attrs["batch_names"]
        columns = g.attrs["omics_names"]
        index_str = g.attrs["batch_names_"]
        columns_str = g.attrs["omics_names_"]
        batch_dims, omics_dims = g.attrs["batch_dims"], g.attrs["omics_dims"]

        grid = ct.OrderedDict()
        for ri, ri_s in zip(index, index_str):
            g_ri, grid_ri = g[ri_s], ct.OrderedDict()
            for ci, ci_s in zip(columns, columns_str):
                if ci_s not in g_ri:
                    grid_ri[ci] = None
                    continue

                h5_datai = g_ri[ci_s]
                if isinstance(h5_datai, Group):
                    # sparse array
                    arr = read_sparse_in_h5(h5_datai)
                    # nr, nc = arr.shape
                elif isinstance(h5_datai, Dataset):
                    # if h5_datai.shape == (2,):
                    #     # 2-tuple shape
                    #     nr, nc = h5_datai[:]
                    #     arr = None
                    # else:
                    # ndarray
                    arr = h5_datai[:]
                    # nr, nc = arr.shape

                grid_ri[ci] = arr
                # # TODO: 会重复赋值
                # omics_dims[j] = nc
                # batch_dims[i] = nr

            grid[ri] = grid_ri

        return DataGrid(grid, index, columns, batch_dims, omics_dims)

    def apply(
        self,
        func: Union[
            Callable[[str, str, typ.DATA_ELEM], typ.DATA_ELEM],
            Callable[[typ.DATA_ELEM], typ.DATA_ELEM],
        ],
    ) -> "DataGrid":
        nsig = len(inspect.signature(func).parameters)
        assert nsig in [
            1,
            3,
        ], "func used by DataGrid.apply must be 1-param or 3-param"
        res = ct.OrderedDict()
        for bk, dats in self._grid.items():
            resi = ct.OrderedDict()
            for ok, dati in dats.items():
                if nsig == 3:
                    new_dati = func(bk, ok, dati)
                else:
                    new_dati = func(dati)
                resi[ok] = new_dati
            res[bk] = resi
        # omics dims可以和原来不同，但是omics_names必须一致
        return DataGrid(
            res, self._batch_names, self._omics_names, self._batch_dims, None
        )
