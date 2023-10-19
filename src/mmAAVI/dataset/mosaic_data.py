# import collections as ct
import collections.abc as cta
import warnings
from copy import deepcopy
from typing import Any, Mapping, Optional, OrderedDict, Sequence, Tuple, Union

import h5py
# import torch
import numpy as np
import pandas as pd
# import scipy.sparse as sp
from anndata import AnnData
from sklearn.model_selection import train_test_split

from .. import typehint as typ
from .data_grid import DataGrid
from .nets_dict import Nets
from .reps_dict import Reps


class MosaicData:
    def __init__(
        self,
        data: Union[DataGrid, typ.DATA_GRID_INIT],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        nets: Mapping[str, DataGrid] = {},
        reps: Mapping[str, DataGrid] = {},
        uns: Mapping[str, Any] = {},
    ) -> None:
        assert isinstance(nets, cta.Mapping), "nets must be a mapping"
        assert isinstance(reps, cta.Mapping), "reps must be a mapping"

        # 处理X
        if not isinstance(data, DataGrid):
            data = DataGrid(data)
        self.X = data

        # 处理Meta dataframe
        # when split the dataset, the blabel codes will be consistent.
        if obs is None:
            batch_ind = pd.Categorical(
                np.repeat(self.X.batch_names, self.X.batch_dims)
            )
            self._obs = pd.DataFrame({"_batch": batch_ind})
        elif "_batch" not in obs.columns:
            batch_ind = pd.Categorical(
                np.repeat(self.X.batch_names, self.X.batch_dims)
            )
            self._obs = obs
            self._obs["_batch"] = batch_ind
        else:
            self._obs = obs

        self._var = var
        omics_ind = pd.Categorical(
            np.repeat(self.X.omics_names, self.X.omics_dims)
        )
        if var is None:
            self._var = pd.DataFrame({"_omics": omics_ind})
        else:
            self._var["_omics"] = omics_ind

        # 处理nets
        self._nets = Nets(nets).link(self)

        # 处理reps
        self._reps = Reps(reps).link(self)

        # 处理uns
        self.uns = uns

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, value) -> pd.DataFrame:
        assert isinstance(value, pd.DataFrame), "obs must be a dataframe"
        assert (
            value.shape[0] == self.X.nobs
        ), "the length of obs must be equal to X.nobs"
        if "_batch" in value.columns:
            warnings.warn("_batch columns existed will be overwrited.")
        new_obs = value.copy()
        new_obs["_batch"] = self._obs["_batch"].values  # 避免index的影响
        self._obs = new_obs

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @var.setter
    def var(self, value) -> pd.DataFrame:
        assert isinstance(value, pd.DataFrame), "var must be a dataframe"
        assert (
            value.shape[0] == self.X.nvar
        ), "the length of var must be equal to X.nvar"
        if "_omics" in value.columns:
            warnings.warn("_omics columns existed will be overwrited.")
        new_var = value.copy()
        new_var["_omics"] = self._var["_omics"].values  # 避免index的影响
        self._var = new_var

    @property
    def nets(self):
        return self._nets

    @nets.setter
    def nets(self, value):
        assert isinstance(
            value, Mapping
        ), "nets must be a mappings of data_grid"
        self._nets = Nets(value).link(self)

    @property
    def reps(self):
        """The reps property."""
        return self._reps

    @reps.setter
    def reps(self, value):
        assert isinstance(
            value, Mapping
        ), "reps must be a mappings of data_grid"
        self._reps = Reps(value).link(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.X.shape

    @property
    def nobs(self) -> int:
        return self.X._nobs

    @property
    def nvar(self) -> int:
        return self.X._nvar

    @property
    def nbatch(self) -> int:
        return self.X._nbatch

    @property
    def nomics(self) -> int:
        return self.X._nomics

    @property
    def batch_dims(self) -> Sequence[int]:
        return self.X._batch_dims

    @property
    def omics_dims(self) -> Sequence[int]:
        return self.X._omics_dims

    @property
    def batch_names(self) -> Sequence[str]:
        return self.X._batch_names

    @batch_names.setter
    def batch_names(self, value):
        self._obs.replace(
            {"_batch": {k: v for k, v in zip(self.X._batch_names, value)}},
            inplace=True,
        )
        for repi in self._reps.values():
            repi.batch_names = value
        self.X.batch_names = value

    @property
    def omics_names(self) -> Sequence[str]:
        return self.X._omics_names

    @omics_names.setter
    def omics_names(self, value):
        self._var.replace(
            {"_omics": {k: v for k, v in zip(self.X._omics_names, value)}},
            inplace=True,
        )
        for repi in self._reps.values():
            repi.omics_names = value
        self.X.omics_names = value
        for neti in self._nets.values():
            neti.batch_names = value
            neti.omics_names = value

    @property
    def batch_dims_dict(self) -> OrderedDict[str, int]:
        return self.X._batch_dims_dict

    @property
    def omics_dims_dict(self) -> OrderedDict[str, int]:
        return self.X._omics_dims_dict

    def __str__(self) -> str:
        msg = str(self.X)
        msg += "obs: %s\n" % ",".join(self._obs.columns.values)
        msg += "var: %s\n" % ",".join(self._var.columns.values)
        if self._nets:
            msg += "nets: %s\n" % ",".join(self._nets.keys())
        if self._reps:
            msg += "reps: %s\n" % ",".join(self._reps.keys())
        if self.uns:
            msg += "uns: %s\n" % ",".join(self.uns.keys())
        return msg

    def sparse_array2matrix(self) -> None:
        self.X.as_sparse_matrix()
        for neti in self.nets.values():
            neti.as_sparse_matrix()

    # def remove_block(
    #     self, batch: typ.GRID_INDEX_ONE, omic: typ.GRID_INDEX_ONE
    # ) -> "MosaicMultiOmicsDataset":
    #     grid = self.X.grid.copy()
    #     grid[batch][omic] = None
    #     return MosaicMultiOmicsDataset(
    #         grid, self._obs, self._var, self._varp, self.reps, self.uns
    #     )

    def save(self, fn: str):
        with h5py.File(fn, mode="w") as h5:
            h5_grid = h5.create_group("X")
            self.X.save_in_h5group(h5_grid)

            if self._reps:
                grep = h5.create_group("reps")
                for k, rep in self._reps.items():
                    rep.save_in_h5group(grep.create_group(k))

            if self._nets:
                h5_net = h5.create_group("nets")
                for k, neti in self._nets.items():
                    neti.save_in_h5group(h5_net.create_group(k))

            # h5.attrs["uns"] = self.uns

        self._obs.to_hdf(fn, "obs", mode="a", format="table")
        self._var.to_hdf(fn, "var", mode="a", format="table")

    @classmethod
    def load(self, fn: str) -> "MosaicData":
        with h5py.File(fn, "r") as h5:
            X = DataGrid.load_from_h5group(h5["X"])

            reps = {}
            if "reps" in h5:
                grep = h5["reps"]
                for k, rep in grep.items():
                    reps[k] = DataGrid.load_from_h5group(rep)

            nets = {}
            if "nets" in h5:
                gnet = h5["nets"]
                for k, neti in gnet.items():
                    nets[k] = DataGrid.load_from_h5group(neti)

        obs = pd.read_hdf(fn, "obs")
        var = pd.read_hdf(fn, "var")

        return MosaicData(X, obs, var, nets, reps)  # , uns=h5.attrs["uns"])

    def to_anndata(
        self,
        sparse: bool = True,
        drop_uns: Optional[Union[Sequence[str], str, bool]] = None,
    ) -> AnnData:
        """drop_uns=True表示将所有的uns都丢弃"""
        X = self.X.to_array(sparse)
        ann = AnnData(X, obs=self._obs, var=self._var)

        for k, v in self._reps.items():
            ann.obsm[k] = v.to_array(False)

        self._nets.expand_full_names()  # 将所有的特征补齐
        for k, v in self._nets.items():
            ann.varp[k] = v.to_array(True)

        if drop_uns is True:
            return ann

        if drop_uns is None or drop_uns is False:
            ann.uns = self.uns
        else:
            if isinstance(drop_uns, str):
                drop_uns = [drop_uns]
            for k, v in self.uns.items():
                if k in drop_uns:
                    continue
                ann.uns[k] = v

        return ann

    def set_null_for_zero_block(self, zero_dim: bool = False):
        self.X.set_null_for_zero_block(zero_dim)

        for repi in self._reps.values():
            repi.set_null_for_zero_block(zero_dim)

    """ partition of mosaic dataset """

    def select_rows_by_batch(
        self, inds: dict[str, Sequence[int]]
    ) -> "MosaicData":
        X = self.X.select_rows_by_batch(inds)
        reps = {
            k: gridi.select_rows_by_batch(inds)
            for k, gridi in self._reps.items()
        }
        obs = pd.concat(
            [
                self._obs.query("_batch == '%s'" % k).iloc[inds[k], :]
                for k in self.batch_names
                if k in inds
            ]
        )
        return MosaicData(
            X,
            obs,
            deepcopy(self._var),
            deepcopy(self._nets),
            reps,
            deepcopy(self.uns),
        )

    def select_rows(self, ind: Union[str, np.ndarray]) -> "MosaicData":
        if isinstance(ind, str):
            ind = self._obs.loc[:, ind].values
        elif not isinstance(ind, np.ndarray):
            raise ValueError(
                "select_rows only accepts name of obs columns or np.ndarray"
            )
        if ind.dtype == "bool":
            assert ind.shape[0] == self.nobs, (
                "the boolean ndarray that is accepted "
                "by select_rows must has the same length as nobs"
            )
            ind = ind.nonzero()[0]
        # check ind is sorted
        assert np.all(
            (ind[1:] - ind[:-1]) >= 0
        ), "the indice of select_rows must be sorted"

        ind_dict, start = {}, 0
        for k, n in self.batch_dims_dict.items():
            end = start + n
            ind_k = ind[(ind >= start) & (ind < end)] - start
            if ind_k.shape[0] > 0:
                ind_dict[k] = ind_k
            start = end
        # TODO: 下面是之前的实现存在问题，可能会导致之前计算的结果是错的
        # 有时间检查一下
        # ind_dict, start = {}, 0
        # ends = np.searchsorted(ind, self.batch_dims)
        # ends = np.r_[0, np.cumsum(ends)]
        # for i, (k, n) in enumerate(self.batch_dims_dict.items()):
        #     indi = ind[ends[i]:ends[i + 1]]
        #     if len(indi) > 0:
        #         indi -= start
        #         ind_dict[k] = indi
        #     start += n
        return self.select_rows_by_batch(ind_dict)

    def split(
        self,
        test_size: float,
        strat_meta: Optional[str] = None,
        seed: typ.SEED = None,
    ) -> Tuple["MosaicData", "MosaicData"]:
        tr_inds, te_inds = {}, {}
        for bk, nrows in self.X.batch_dims_dict.items():
            obsi = self._obs.query("_batch == '%s'" % bk)
            strati = None if strat_meta is None else obsi[strat_meta].values
            tr_ind, te_ind = train_test_split(
                np.arange(nrows),
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=strati,
            )
            tr_inds[bk] = tr_ind
            te_inds[bk] = te_ind

        return (
            self.select_rows_by_batch(tr_inds),
            self.select_rows_by_batch(te_inds),
        )

    # def resample(self, ratio: dict[str, float], replace: bool = True):
    #     if any(r > 1 for r in ratio.values()):
    #         assert replace, "ratio > 1, must be replaced sampling"
    #
    #     inds = {}
    #     for k, ni in self.batch_dims_dict.items():
    #         ri = ratio.get(k, 1.)
    #         if ri == 1.0:
    #             ind = np.arange(ni)
    #         else:
    #             ind = np.random.choice(ni, size=int(ni*ri), replace=replace)
    #         inds[k] = ind
    #
    #     return self.select_rows_by_batch(inds)
