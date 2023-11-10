import bisect
import collections as ct
import logging
import random
from typing import Any, Callable, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data as D

from ..typehint import BATCH, SLICE, TUPLE_NET, T  # , NETS
from .mosaic_data import MosaicData
from .utils import normalize_edges, sample_negs1, sample_negs2


class PairBatchRandomSampler:
    def __init__(
        self,
        blabel1: np.ndarray,
        blabel2: np.ndarray,
        nsamples: int,
        dat_ind: int = 1,
        seed: int = 0,
    ) -> None:
        assert dat_ind in [1, 2]

        batch1_uni = np.unique(blabel1)
        batch2_uni = np.unique(blabel2)
        self.blabel_uni = np.intersect1d(batch1_uni, batch2_uni)

        # preprocess batch indices
        self.batch1_dict = {
            k: np.nonzero(blabel1 == k)[0] for k in self.blabel_uni
        }
        self.batch2_dict = {
            k: np.nonzero(blabel2 == k)[0] for k in self.blabel_uni
        }

        # calculate the probabilities of each batch
        cnt = pd.value_counts(np.concatenate([blabel1, blabel2]))
        cnt = cnt.loc[self.blabel_uni]
        cnt /= cnt.sum()

        # calculate the
        self.bratio = cnt.values
        self.nsamples = nsamples
        self.dat_ind = dat_ind
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        bsamples = self.rng.multinomial(self.nsamples, self.bratio)
        blabel_dict = (
            self.batch1_dict if self.dat_ind == 1 else self.batch2_dict
        )
        res = []
        for k, bni in zip(self.blabel_uni, bsamples):
            resi = self.rng.choice(blabel_dict[k], bni)
            res.append(resi)
        res = np.concatenate(res)
        return iter(res)

    def __len__(self):
        return self.nsamples


class ParallelDataLoader:
    r"""
    Parallel data loader

    Parameters
    ----------
    *data_loaders
        An arbitrary number of data loaders，yield dict
    cycle_flags
        Whether each data loader should be cycled in case they are of
        different lengths, by default none of them are cycled.
    """

    def __init__(
        self,
        *data_loaders: D.DataLoader,
        cycle_flags: Optional[List[bool]] = None
    ) -> None:
        cycle_flags = cycle_flags or [False] * len(data_loaders)
        if len(cycle_flags) != len(data_loaders):
            raise ValueError("Invalid cycle flags!")
        self.cycle_flags = cycle_flags
        self.data_loaders = list(data_loaders)
        self.num_loaders = len(self.data_loaders)
        self.iterators = None

    def __len__(self):
        return min(
            len(dati)
            for dati, flagi in zip(self.data_loaders, self.cycle_flags)
            if not flagi
        )

    def __iter__(self) -> "ParallelDataLoader":
        self.iterators = [iter(loader) for loader in self.data_loaders]
        return self

    def _next(self, i: int) -> BATCH:
        try:
            return next(self.iterators[i])
        except StopIteration as e:
            if self.cycle_flags[i]:
                self.iterators[i] = iter(self.data_loaders[i])
                return next(self.iterators[i])
            raise e

    def __next__(self) -> BATCH:
        res = {}
        for i in range(self.num_loaders):
            res.update(self._next(i))
        return res


class GraphDataLoader:
    def __init__(
        self,
        nets: TUPLE_NET,
        net_style: Literal["dict", "sarr", "walk"],
        batch_size: Optional[Union[float, int]] = None,
        num_workers: Optional[int] = 0,
        drop_self_loop: bool = True,
        walk_length: int = 20,
        context_size: Optional[int] = 10,
        walks_per_node: int = 10,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
    ):
        if net_style != "sarr":
            raise NotImplementedError
        assert batch_size is not None
        assert nets[0][0] == nets[0][1], "the dims of network must be equal"

        self._index = None

        self.nets = nets
        self.net_style = net_style
        self.n = np.inf  # 如果没有设置n，则n是无穷大
        self.bs = batch_size
        self.nw = num_workers
        self.drop_self_loop = drop_self_loop
        self.vnum = nets[0][0]
        self.nns = num_negative_samples
        logging.info("number of edges in network is %d" % (nets[1].shape[0]))

        if isinstance(self.bs, float):
            if self.net_style == "walk":
                self.bs_int = int(self.bs * self.vnum)
            elif self.net_style == "sarr":
                if self.drop_self_loop:
                    n_edges = (self.nets[1] != self.nets[2]).sum()
                else:
                    n_edges = self.nets[1].shape[0]
                # 注意还有负采样
                self.bs_int = int(n_edges * (1 + self.nns) * self.bs)
        else:
            self.bs_int = self.bs
        logging.info("network batch size is %d" % self.bs_int)

        if self.net_style == "walk":
            pass
            # self.walk_length = walk_length
            # self.context_size = context_size
            # self.walks_per_node = walks_per_node
            # self.num_negative_samples = num_negative_samples
            # self.p = p
            # self.q = q
            # self.random_walk_fn = torch.ops.torch_cluster.random_walk
            #
            # if self.context_size is not None:
            #     self.num_walks_per_rw = (
            #         1 + self.walk_length + 1 - self.context_size
            #     )
            #
            # edge_index = torch.tensor(
            #     np.stack([nets[1], nets[2]], axis=0), dtype=torch.long
            # )
            # row, col = sort_edge_index(edge_index, num_nodes=self.vnum).cpu()
            # self.walk_need = index2ptr(row, self.vnum), col
            # self.net_sign = torch.tensor(self.nets[3]).float()
            #
            # self.walk_loader = D.DataLoader(
            #     range(self.vnum),
            #     batch_size=self.bs_int,
            #     shuffle=True,
            #     num_workers=self.nw,
            #     collate_fn=self.sample_rw,
            # )

    def __iter__(self):
        if self.net_style == "walk":
            # def generator():
            #     for walk in self.sampled_loader:
            #         yield {"walk": walk, "coo": self.sarr_sign}
            # return generator()
            return iter(self.walk_loader)
        if self.net_style == "sarr":
            self.sample_graph()
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.n:
            raise StopIteration
        if self.net_style == "dict":
            net = {}
            for k, sarr in self.nets.items():
                negi, negj = sample_negs1(sarr, sarr.nnz)
                posi, posj, wt = sarr.row, sarr.col, sarr.data
                esign = (wt > 0).astype(float)
                wt = np.abs(wt)

                iall = torch.tensor(
                    np.concatenate([posi, negi]), dtype=torch.long
                )
                jall = torch.tensor(
                    np.concatenate([posj, negj]), dtype=torch.long
                )
                sall = torch.tensor(
                    np.concatenate([esign, np.ones_like(esign)]),
                    dtype=torch.float32,
                )
                wall = torch.tensor(
                    np.concatenate([wt, np.zeros_like(wt)]),
                    dtype=torch.float32,
                )
                ind = torch.randperm(iall.size(0))
                net[k] = (iall[ind], jall[ind], sall[ind], wall[ind])
            return {"varp": net}
        elif self.net_style == "sarr":
            index = slice(
                self._index * self.bs_int, (self._index + 1) * self.bs_int
            )
            sample_graph = tuple(a[index] for a in self.sampled_net)
            return {"sample_graph": sample_graph, "varp": self.sampled_net}
        elif self.net_style == "walk":
            raise NotImplementedError  # walk使用其他的迭代器

        self._index += 1

    def sample_graph(self) -> None:
        iall, jall, sall, wall = sample_negs2(
            self.nets, neg_samples=self.nns, drop_self_loop=self.drop_self_loop
        )
        enorm = normalize_edges(self.nets)
        enorm_all = torch.tensor(
            np.concatenate([enorm, np.zeros(len(iall) - len(enorm))]),
            dtype=torch.float32,
        )
        iall = torch.tensor(iall, dtype=torch.long)
        jall = torch.tensor(jall, dtype=torch.long)
        sall = torch.tensor(sall, dtype=torch.float32)
        wall = torch.tensor(wall, dtype=torch.float32)
        ind = torch.randperm(iall.size(0))
        self.sampled_net = (
            iall[ind],
            jall[ind],
            sall[ind],
            wall[ind],
            enorm_all[ind],
        )
        self.n_sample_net = self.sampled_net[0].shape[0]
        self.n = (self.n_sample_net + self.bs_int - 1) // self.bs_int

    def sample_rw(self, batch: T):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        # pos_sample
        start = batch.repeat(self.walks_per_node)
        pos_rw, ed_ind = self.random_walk_fn(
            *self.walk_need,
            start,
            self.walk_length,
            self.p,
            self.q,
        )
        pos_sign = self.net_sign[ed_ind]

        # neg_sample
        start = batch.repeat(self.walks_per_node * self.num_negative_samples)
        neg_rw = torch.randint(
            self.vnum,
            (start.size(0), self.walk_length),
            dtype=start.dtype,
            device=start.device,
        )
        neg_rw = torch.cat([start.view(-1, 1), neg_rw], dim=-1)

        if self.context_size is not None:
            pos_rw_s, neg_rw_s, pos_sign_s = [], [], []
            for j in range(self.num_walks_per_rw):
                pos_rw_s.append(pos_rw[:, j:(j + self.context_size)])
                neg_rw_s.append(neg_rw[:, j:(j + self.context_size)])
                pos_sign_s.append(pos_sign[:, j:(j + self.context_size - 1)])
            pos_rw = torch.cat(pos_rw_s, dim=0)
            neg_rw = torch.cat(neg_rw_s, dim=0)
            pos_sign = torch.cat(pos_sign_s, dim=0)

        return {"walk": (pos_rw, neg_rw), "pos_sign": pos_sign}


class BalanceSizeSampler:
    def __init__(
        self,
        resample_size: dict[str, float],
        blabels: np.ndarray,
    ) -> None:
        self._resample_size = resample_size
        self._batch_index_dict, self._n = {}, 0
        blabel_uni = np.unique(blabels)
        for bi in blabel_uni:
            ind = np.nonzero(blabels == bi)[0]
            self._batch_index_dict[bi] = ind
            self._n += int(ind.shape[0] * self._resample_size.get(bi, 1.0))

    def _get_index(self):
        index = []
        for bk, bi in self._batch_index_dict.items():
            ratio = self._resample_size.get(bk, 1.0)
            if ratio == 1.0:
                index.append(bi)
            else:
                target_size = int(ratio * bi.shape[0])
                index.append(np.random.choice(bi, target_size, replace=True))
        index = np.concatenate(index)
        np.random.shuffle(index)
        return index

    def __iter__(self):
        return iter(self._get_index())

    def __len__(self) -> int:
        return self._n


class SemiSupervisedSampler:

    """
    为半监督生成batch，
        保证每个batch都包含足够的unlabel和labeled样本。
        他们两者是分别采样，然后放在一起的。
        其总批次数量=批次数量较大的那个

    需要同时控制random和np.random的随机性
    """

    def __init__(
        self,
        is_labeled: np.ndarray,
        unlabel_batch_size: int,
        labeled_batch_size: int,
        resample_size: Optional[dict[str, float]] = None,
        blabels: Optional[np.ndarray] = None,
        drop_last: bool = True,
    ) -> None:
        self._is_labeled = is_labeled
        self._ubs = unlabel_batch_size
        self._lbs = labeled_batch_size
        self._flag_resample = resample_size is not None
        self._dl = drop_last

        if self._flag_resample:
            assert blabels is not None
            self._balance_sampler = BalanceSizeSampler(resample_size, blabels)
            self.n = len(self._balance_sampler)
            # TODO: 为了让第一个epoch也有length，但是可能不准
            index = self._balance_sampler._get_index()
            is_labeled_re = self._is_labeled[index]
            n_l = is_labeled_re.sum()
            n_u = self.n - n_l
            if self._dl:
                self.nb_l = n_l // self._lbs
                self.nb_u = n_u // self._ubs
            else:
                self.nb_l = (n_l + self._lbs - 1) // self._lbs
                self.nb_u = (n_u + self._ubs - 1) // self._ubs
            self.nb = max(self.nb_l, self.nb_u)
        else:
            self.n = self._is_labeled.shape[0]
            self.n_l = self._is_labeled.sum()
            self.n_u = self.n - self.n_l
            if drop_last:
                self.nb_l = self.n_l // self._lbs
                self.nb_u = self.n_u // self._ubs
            else:
                self.nb_l = (self.n_l + self._lbs - 1) // self._lbs
                self.nb_u = (self.n_u + self._ubs - 1) // self._ubs
            self.nb = max(self.nb_l, self.nb_u)

            self.inds_l = np.nonzero(self._is_labeled)[0]
            self.inds_u = np.nonzero(np.logical_not(self._is_labeled))[0]

    def _iter_wo_resample(self):
        inds_l = self.inds_l.copy()
        inds_u = self.inds_u.copy()

        for i in range(self.nb):
            li, ui = i % self.nb_l, i % self.nb_u
            if li == 0:
                np.random.shuffle(inds_l)
            if ui == 0:
                np.random.shuffle(inds_u)

            batch_l = inds_l[(li * self._lbs):((li + 1) * self._lbs)]
            batch_u = inds_u[(ui * self._ubs):((ui + 1) * self._ubs)]

            yield np.r_[batch_l, batch_u]

    def _iter_w_resample(self):
        index = self._balance_sampler._get_index()
        is_labeled_re = self._is_labeled[index]
        n_l = is_labeled_re.sum()
        n_u = self.n - n_l
        if self._dl:
            self.nb_l = n_l // self._lbs
            self.nb_u = n_u // self._ubs
        else:
            self.nb_l = (n_l + self._lbs - 1) // self._lbs
            self.nb_u = (n_u + self._ubs - 1) // self._ubs
        self.nb = max(self.nb_l, self.nb_u)

        self.inds_l = np.nonzero(is_labeled_re)[0]
        self.inds_u = np.nonzero(np.logical_not(is_labeled_re))[0]

        for batch in self._iter_wo_resample():
            yield [index[i] for i in batch]

    def __iter__(self):
        if self._flag_resample:
            return self._iter_w_resample()
        else:
            return self._iter_wo_resample()

    def __len__(self):
        return self.nb


class TorchMapDataset(D.Dataset):
    def __init__(
        self,
        data: MosaicData,
        input_use: Optional[str] = None,
        output_use: Optional[str] = None,
        obs_blabel: Optional[str] = None,
        obs_dlabel: Optional[str] = None,
        obs_sslabel: Optional[str] = None,
        obs_sslabel_full: Optional[str] = None,
        impute_miss: bool = False,
        sslabel_codes: Optional[Sequence[Any]] = None,
    ) -> None:
        if input_use is not None:
            assert input_use in data._reps.keys()
        if output_use is not None:
            assert output_use in data._reps.keys()
        # if net_use is not None:
        #     assert net_use in data._nets.keys()
        if obs_sslabel is not None:
            if sslabel_codes is None:
                raise ValueError(
                    "if semi-supervised learning, must "
                    "explictly encode the sslabels."
                )
            sslabel_mapping = {k: i for i, k in enumerate(sslabel_codes)}

        super().__init__()

        self._mdata = data
        self._nrows_cum = np.cumsum(data.batch_dims)
        self._imp_miss = impute_miss

        # prepare input and output ...
        self._in_eq_out = (
            input_use is None and output_use is None
        ) or input_use == output_use
        self._inpt_grid = data.reps[input_use] if input_use else data.X
        self._oupt_grid = data.reps[output_use] if output_use else data.X
        self._inpt_grid.as_sparse_type("csr")
        self._oupt_grid.as_sparse_type("csr")

        # prepare meta variables ...
        self._use_obs = pd.DataFrame(index=data.obs.index)
        # 处理blabel和dlabel
        bl = "_batch" if obs_blabel is None else obs_blabel
        dl = "_batch" if obs_dlabel is None else obs_dlabel
        self._blabel = data.obs[bl].astype("category")
        self._blabel_codes = self._blabel.cat.codes.values
        self._dlabel = data.obs[dl].astype("category")
        self._dlabel_codes = self._dlabel.cat.codes.values
        # 处理sslabel
        if obs_sslabel is not None:
            sslabel = data.obs[obs_sslabel].copy()
            sslabel.replace(sslabel_mapping, inplace=True)
            sslabel.fillna(-1, inplace=True)
            self._sslabel_codes = sslabel.values
            # 记录sslabel的无缺失版本，用于计算指标
            if obs_sslabel_full is not None:
                sslabelf = data.obs[obs_sslabel_full].copy()
                sslabelf.replace(sslabel_mapping, inplace=True)
                self._sslabel_full_codes = sslabelf.values
            else:
                self._sslabel_full_codes = None
        else:
            self._sslabel_codes = None
            self._sslabel_full_codes = None

        self._keys_od = ["input", "output", "mask"]
        if self._imp_miss:
            self._keys_od.append("imp_blabel")
        self._keys_od = np.array(self._keys_od).astype(np.str_)

        self._keys_arr = ["blabel", "dlabel"]
        if self._sslabel_codes is not None:
            self._keys_arr.append("sslabel")
        self._keys_arr = np.array(self._keys_arr).astype(np.str_)

    def __len__(self) -> int:
        return self._mdata.nobs

    def __getitem__(self, index: int) -> SLICE:
        bi = bisect.bisect_right(self._nrows_cum, index)
        bk = self._mdata.batch_names[bi]
        si = index if bi == 0 else index - self._nrows_cum[bi - 1]

        patch = {
            "ind": self._use_obs.index[index],
            "input": ct.OrderedDict(),
            "output": ct.OrderedDict(),
            "mask": ct.OrderedDict(),
            "blabel": self._blabel_codes[index],
            "dlabel": self._dlabel_codes[index],
        }

        # 如果有sslabel，加入到slice中
        if self._sslabel_codes is not None:
            patch["sslabel"] = self._sslabel_codes[index]

        for ok in self._mdata.omics_names:
            oupti = self._oupt_grid[bk, ok, si]
            mask = oupti is not None
            if mask:  # 存在想要的样本
                inpti = (
                    oupti if self._in_eq_out else self._inpt_grid[bk, ok, si]
                )
                if self._imp_miss:
                    patch.setdefault("imp_blabel", ct.OrderedDict())[
                        ok
                    ] = patch["blabel"]

            else:
                if self._imp_miss:
                    bi_imp = random.choice(self._oupt_grid._batch_nonone[ok])
                    si_imp = random.choice(
                        range(self._oupt_grid.batch_dims[bi_imp])
                    )
                    oupti = self._oupt_grid[bi_imp, ok, si_imp]
                    inpti = (
                        oupti
                        if self._in_eq_out
                        else self._inpt_grid[bi_imp, ok, si_imp]
                    )
                    patch.setdefault("imp_blabel", ct.OrderedDict())[
                        ok
                    ] = bi_imp
                else:
                    oupti = torch.zeros(
                        self._oupt_grid.omics_dims_dict[ok],
                        dtype=torch.float32,
                    )
                    inpti = torch.zeros(
                        self._inpt_grid.omics_dims_dict[ok],
                        dtype=torch.float32,
                    )

            patch["input"][ok] = inpti
            patch["output"][ok] = oupti
            patch["mask"][ok] = mask

        return patch

    def get_collated_fn(self) -> Callable[[List[SLICE]], BATCH]:
        def f(batches: List[SLICE]) -> BATCH:
            res = {k: ct.OrderedDict() for k in self._keys_od}
            for batchi in batches:
                for k in self._keys_od:
                    for ok, v in batchi[k].items():
                        res[k].setdefault(ok, []).append(v)

                for k in self._keys_arr:
                    res.setdefault(k, []).append(batchi[k])

            for k in self._keys_od:
                resk = res[k]
                for kk in resk.keys():
                    if k == "mask":
                        resk[kk] = torch.tensor(resk[kk], dtype=torch.float32)
                    else:
                        resk[kk] = torch.tensor(
                            np.stack(resk[kk]), dtype=torch.float32
                        )

            for k in self._keys_arr:
                res[k] = torch.tensor(res[k], dtype=torch.long)

            res["ind"] = [batchi["ind"] for batchi in batches]
            return res

        return f
