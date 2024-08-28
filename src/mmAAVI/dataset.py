import logging
import warnings
from typing import Set, List, Optional, Union, Dict, Tuple, TypedDict, Literal

import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import scipy.sparse as sp
from mudata import MuData

from .utils import to_dense

# from mmAAVI.cy_utils import negative_sample_from_graph


SUBGRAPH = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
NORMED_SUBGRAPH = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


class SAMPLE(TypedDict, total=False):
    input: Dict[str, torch.Tensor]
    output: Dict[str, torch.Tensor]
    mask: Dict[str, torch.Tensor]
    blabel: int
    dlabel: int
    sslabel: int


class MINIBATCH(TypedDict, total=False):
    input: Dict[str, torch.Tensor]
    output: Dict[str, torch.Tensor]
    mask: Dict[str, torch.Tensor]
    blabel: torch.Tensor
    dlabel: torch.Tensor
    sslabel: torch.Tensor


class GMINIBATCH(TypedDict, total=False):
    input: Dict[str, torch.Tensor]
    output: Dict[str, torch.Tensor]
    mask: Dict[str, torch.Tensor]
    blabel: torch.Tensor
    dlabel: torch.Tensor
    sslabel: torch.Tensor
    graph_minibatch: NORMED_SUBGRAPH
    normed_subgraph: NORMED_SUBGRAPH


class MosaicMuDataset(D.Dataset):
    def __init__(
        self,
        mdata: MuData,
        input_key: Optional[str] = "log1p_norm",
        output_key: Optional[str] = None,
        batch_key: str = "batch",
        dlabel_key: Optional[str] = None,
        sslabel_key: Optional[str] = None,
    ) -> None:
        # TODO: how to handle that batch_key is None?
        if batch_key is None:
            raise NotImplementedError

        super().__init__()

        self._mdata = mdata
        self._input_key = input_key
        self._output_key = output_key
        self._batch_key = batch_key
        self._dlabel_key = dlabel_key if dlabel_key is not None else batch_key
        self._sslabel_key = sslabel_key

        self._input_ndims = {
            k: (
                adatai.X.shape[1]
                if self._input_key is None
                else adatai.obsm[self._input_key].shape[1]
            )
            for k, adatai in self._mdata.mod.items()
        }
        self._output_ndims = {
            k: (
                adatai.X.shape[1]
                if self._output_key is None
                else adatai.obsm[self._output_key].shape[1]
            )
            for k, adatai in self._mdata.mod.items()
        }

    def __len__(self) -> int:
        return self._mdata.n_obs

    def __getitem__(self, index: int) -> SAMPLE:
        row = self._mdata[index, :]
        inpts, outputs, mask = {}, {}, {}
        for k, row_ann in row.mod.items():
            # use bool to convert the values with np.bool_,
            # avoid deprecated warning
            mask[k] = bool(self._mdata.obsm[k][index])
            # NOTE: can not use row_ann.n_obs==0 to prove the omic missing,
            # it is also happened when count is 0.
            # the true approach is by mudata.obsm["OMIC_NAME"][i]
            if row_ann.n_obs == 0:
                inpts[k] = torch.zeros(
                    self._input_ndims[k], dtype=torch.float32
                )
                outputs[k] = torch.zeros(
                    self._output_ndims[k], dtype=torch.float32
                )
            else:
                inpts[k] = torch.tensor(
                    to_dense(
                        row_ann.X[0, :]
                        if self._input_key is None
                        else row_ann.obsm[self._input_key]
                    ),
                    dtype=torch.float32,
                )
                outputs[k] = torch.tensor(
                    to_dense(
                        row_ann.X[0, :]
                        if self._output_key is None
                        else row_ann.obsm[self._output_key]
                    ),
                    dtype=torch.float32,
                )

        patch = {
            "input": inpts,
            "output": outputs,
            "mask": mask,
            "blabel": self._mdata.obs[self._batch_key].iloc[index],
            "dlabel": self._mdata.obs[self._dlabel_key].iloc[index],
        }
        if self._sslabel_key is not None:
            patch["sslabel"] = self._mdata.obs[self._sslabel_key].iloc[index]
        return patch


class BalanceSizeSampler:
    """
    Adjust the sampling ratio according to a specific label (such as batch) to
    ensure that the number of these labels remains consistent during training.
    You cannot directly resample MuData and then use it in the loader
    because we must ensure that the samples resampled for each epoch
    are different to ensure generalization.
    """

    def __init__(
        self,
        mdata: MuData,
        label_key: str,
        sample_size: Union[str, int] = "max",
    ) -> None:
        if sample_size not in ["max", "min", "mean"]:
            assert isinstance(
                sample_size, int
            ), "sample ratio must be min, max or integer."

        label_arr = mdata.obs[label_key].values
        label_unique = np.unique(label_arr)
        self._label_len = len(label_unique)

        self._indices_list = [
            np.nonzero(label_arr == i)[0] for i in label_unique
        ]
        self._n_true = [len(ind_arr) for ind_arr in self._indices_list]

        if sample_size == "max":
            self._n_balance = max(self._n_true)
        elif sample_size == "min":
            self._n_balance = min(self._n_true)
        elif sample_size == "mean":
            self._n_balance = int(np.mean(self._n_true))
        else:
            self._n_balance = sample_size

    def __iter__(self):
        index = []
        for arr in self._indices_list:
            index.append(np.random.choice(arr, self._n_balance, replace=True))
        index = np.concatenate(index)
        np.random.shuffle(index)
        return iter(index)

    def __len__(self) -> int:
        return self._label_len * self._n_balance


class SemiSupervisedSampler:
    """
    Note, this is a batch sampler.

    For semi-supervised batch generation,
    Ensure each batch contains enough unlabeled and labeled samples.
    They are sampled separately and then combined together.
    The total number of batches equals the larger batch number.
    You need to control the randomness of both random and np.random.
    """

    def __init__(
        self,
        mdata: MuData,
        sslabel_key: str,
        nan_as: int = -1,
        batch_size: int = 256,
        label_ratio: float = 0.2,
        drop_last: bool = False,
        shuffle: bool = True,
        repeat_sample: bool = True,
        balance_label_key: Optional[str] = None,
        balance_sample_size: Union[str, int] = "max",
    ) -> None:
        """
        label_ratio indicates the proportion of labeled samples
            in a batch size.
        shuffle=True only works when balance_label_key is None.
        repeat_sample=True ensures that if there are not enough labeled or
            unlabeled samples, resampling starts from the beginning, ensuring
            each batch contains both labeled and unlabeled samples.
        """

        assert (label_ratio >= 0.0) and (
            label_ratio <= 1.0
        ), "label_ratio must be in [0, 1]."

        if balance_label_key is not None:
            assert shuffle, "shuffle must be true if use balance_label_key."

        self._balance_label_key = balance_label_key
        self._shuffle = shuffle
        self._repeat_sample = repeat_sample

        self._label_bs = int(
            batch_size * label_ratio
        )  # the number of label samples in certain batch
        self._unlabel_bs = batch_size - self._label_bs

        sslabel = mdata.obs[sslabel_key].values
        is_unlabel = sslabel == nan_as

        if balance_label_key is not None:
            balance_sampler = BalanceSizeSampler(
                mdata, balance_label_key, balance_sample_size
            )
            n_balance = balance_sampler._n_balance

            balance_label = mdata.obs[balance_label_key]
            ctab = pd.crosstab(is_unlabel, balance_label, margins=True)
            self._ctab_balance = (
                (ctab * (n_balance / ctab.loc["All", :].values))
                .round(0)
                .astype(int)
            )
            size_label, size_unlabel = (
                self._ctab_balance.iloc[:-1, :-1].sum(axis=1).tolist()
            )

            batch_array = mdata.obs[balance_label_key].values
            batch_unique = np.unique(batch_array)
            self._indices_dict = {"label": {}, "unlabel": {}}
            for batchi in batch_unique:
                batch_mask = batch_array == batchi
                self._indices_dict["label"][batchi] = np.nonzero(
                    batch_mask & (~is_unlabel)
                )[0]
                self._indices_dict["unlabel"][batchi] = np.nonzero(
                    batch_mask & (is_unlabel)
                )[0]
        else:
            # indices of labelled samples
            self._indices_unlabel = np.nonzero(is_unlabel)[0]
            # indices of unlabelled samples
            self._indices_label = np.nonzero(np.logical_not(is_unlabel))[0]
            # number of labelled samples
            size_unlabel = self._indices_unlabel.shape[0]
            # number of unnlabelled samples
            size_label = self._indices_label.shape[0]

        if drop_last:
            self._num_batch_unlabel = size_unlabel // self._unlabel_bs
            self._num_batch_label = size_label // self._label_bs
        else:
            self._num_batch_unlabel = (
                size_unlabel + self._unlabel_bs - 1
            ) // self._unlabel_bs
            self._num_batch_label = (
                size_label + self._label_bs - 1
            ) // self._label_bs
        self._num_batch = max(self._num_batch_label, self._num_batch_unlabel)

    def _iter_wo_balance(self):
        inds_l = self._indices_label.copy()
        inds_u = self._indices_unlabel.copy()

        for i in range(self._num_batch):
            if i == 0 and self._shuffle:
                np.random.shuffle(inds_l)
                np.random.shuffle(inds_u)

            if i < self._num_batch_label:
                batch_l = inds_l[
                    (i * self._label_bs) : ((i + 1) * self._label_bs)
                ]
            elif self._repeat_sample:
                li = i % self._num_batch_label
                if li == 0 and self._shuffle:
                    np.random.shuffle(inds_l)
                batch_l = inds_l[
                    (li * self._label_bs) : ((li + 1) * self._label_bs)
                ]
            else:
                # NOTE: Do not use []. np.r_ can still work, but it will
                # default to converting dtype from int to float, which will
                # fail when used as an index in getitem.
                batch_l = np.array([], dtype=int)

            if i < self._num_batch_unlabel:
                batch_u = inds_u[
                    (i * self._unlabel_bs) : ((i + 1) * self._unlabel_bs)
                ]
            elif self._repeat_sample:
                ui = i % self._num_batch_unlabel
                if ui == 0 and self._shuffle:
                    np.random.shuffle(inds_u)
                batch_u = inds_u[
                    (ui * self._unlabel_bs) : ((ui + 1) * self._unlabel_bs)
                ]
            else:
                batch_u = np.array([], dtype=int)

            yield np.r_[batch_l, batch_u]

    def _iter_w_balance(self):
        # sample from indices_dict by ctab_balance，construct initial indices
        # label
        self._indices_label = []
        for k, indices_k in self._indices_dict["label"].items():
            target_size = self._ctab_balance.loc[False, k]
            if target_size > 0:
                self._indices_label.append(
                    np.random.choice(indices_k, target_size, replace=True)
                )
        self._indices_label = np.concatenate(self._indices_label)

        self._indices_unlabel = []
        for k, indices_k in self._indices_dict["unlabel"].items():
            target_size = self._ctab_balance.loc[True, k]
            if target_size > 0:
                self._indices_unlabel.append(
                    np.random.choice(indices_k, target_size, replace=True)
                )
        self._indices_unlabel = np.concatenate(self._indices_unlabel)

        for batch in self._iter_wo_balance():
            yield batch

    def __iter__(self):
        if self._balance_label_key is None:
            return self._iter_wo_balance()
        else:
            return self._iter_w_balance()

    def __len__(self):
        return self._num_batch


class GraphDataLoader:
    @staticmethod
    def vertex_degrees(
        net_abs: Union[sp.csr_array, sp.coo_array], direction: str = "both"
    ):
        if direction == "in":
            return net_abs.sum(axis=0)
        elif direction == "out":
            return net_abs.sum(axis=1)
        elif direction == "both":
            return (
                net_abs.sum(axis=0) + net_abs.sum(axis=1) - net_abs.diagonal()
            )
        raise ValueError("Unrecognized direction!")

    @staticmethod
    def negative_sample_from_graph_1(
        vnum: int, eset: Set, i_neg: np.ndarray, vprob: np.ndarray
    ) -> np.ndarray:
        """
        Perform negative sampling from the graph, where the starting point of
        the sampled edge is already determined (i_neg), but the endpoint is
        not yet determined.

        During the sampling process, it is necessary to ensure that the
        probability of the endpoint being sampled follows a given probability.

        The graph will only use its starting and ending coordinates, the edge
        values will not be used, so there is no need to perform abs and binary.

        This implementation may result in an infinity loop.
        """
        j_neg = np.random.choice(vnum, i_neg.size, replace=True, p=vprob)
        # maybe some negative edge is actually pos, remove and re-sampling
        remain = np.where([item in eset for item in zip(i_neg, j_neg)])[0]
        while remain.size:  # NOTE: Potential infinite loop if graph too dense
            newnj = np.random.choice(vnum, remain.size, replace=True, p=vprob)
            j_neg[remain] = newnj
            remain = remain[
                [item in eset for item in zip(i_neg[remain], newnj)]
            ]
        return j_neg

    @staticmethod
    def negative_sample_from_graph_2(
        graph: sp.csr_matrix, i_neg: np.ndarray, vprob: np.ndarray
    ) -> np.ndarray:
        """
        This implementation samples for each starting point in a loop, and
        only samples from its non-neighboring nodes.
        """
        indptr, indices = graph.indptr, graph.indices
        all_indices = np.arange(graph.shape[0])
        res = []
        for i in i_neg:
            cind_i = indices[indptr[i] : indptr[i + 1]]
            cind_i_neg = np.setdiff1d(all_indices, cind_i)
            vprob_i_neg = vprob[cind_i_neg]
            j_i_neg = np.random.choice(
                cind_i_neg, p=vprob_i_neg / vprob_i_neg.sum()
            )
            res.append(j_i_neg)
        return np.array(res)

    @staticmethod
    def sample_subgraph(
        net: sp.csr_array,
        neg_samples: int = 1,
        weighted_sampling: bool = True,
        drop_self_loop: bool = True,
    ) -> SUBGRAPH:
        # sample whole graph, not sample subgraph.
        # this can be seen as add noise into the graph.
        vnum = net.shape[0]

        # NOTE: 遵照之前的写法，把drop_self_loop前置，其会影响到i\j\es\ew的值
        # NOTE: Following the previous writing, move drop_self_loop to an
        # earlier stage, as it will affect the values of i, j, es, and ew.
        # TODO: This is a bit different from what I originally wrote.
        # Here, setdiag will affect the calculation of the degree; however,
        # the original approach did not affect the calculation of the degree.
        if drop_self_loop:
            net.setdiag(0)

        net_coo = net.tocoo()
        i, j, ew = net_coo.row, net_coo.col, net_coo.data
        es = 2 * (ew > 0) - 1
        ew = np.abs(ew)
        eset = set(zip(i, j))

        # shape, i, j, es, ew = tuple_net
        # assert shape[0] == shape[1], "just suitable symmetric matrix"
        # vnum = shape[0]
        # # 去掉self-loop
        # if drop_self_loop:
        #     ind = i != j
        #     i, j, es, ew = i[ind], j[ind], es[ind], ew[ind]

        # == calculate probability of vertices
        degree = (
            GraphDataLoader.vertex_degrees(abs(net))
            if weighted_sampling
            else np.ones(vnum, dtype=float)
        )
        # NOTE: == code is not modified ==
        degree_sum = degree.sum()
        if degree_sum:
            vprob = degree / degree_sum  # Vertex sampling probability
        else:  # Possible when `deemphasize_loops` is set on a loop-only graph
            vprob = np.fill(vnum, 1 / vnum)
        # NOTE: ========================

        # == calculate probability of edges
        effective_enum = ew.sum()
        eprob = ew / effective_enum  # Edge sampling probability
        effective_enum = round(effective_enum)

        # == sample the positive edges
        edge_ind_pos = np.random.choice(
            ew.size, effective_enum, replace=True, p=eprob
        )
        i_pos, j_pos = i[edge_ind_pos], j[edge_ind_pos]
        ew_pos = np.ones_like(i_pos, dtype=np.float32)
        es_pos = es[edge_ind_pos]

        # == sample the negative edges, prefer bigger vertex prob
        i_neg = np.tile(i_pos, neg_samples)
        ew_neg = np.zeros(effective_enum * neg_samples, dtype=np.float32)
        es_neg = np.tile(es_pos, neg_samples)
        j_neg = GraphDataLoader.negative_sample_from_graph_1(
            vnum, eset, i_neg, vprob
        )

        i_sample = np.concatenate([i_pos, i_neg])
        j_sample = np.concatenate([j_pos, j_neg])
        ew_sample = np.concatenate([ew_pos, ew_neg])
        es_sample = np.concatenate([es_pos, es_neg])

        # perm = np.random.permutation(iall.shape[0])
        # return iall[perm], jall[perm], sall[perm], wall[perm]
        return (vnum, i_sample, j_sample, ew_sample, es_sample)

    @staticmethod
    def normalize_edges(
        subgraph: SUBGRAPH, method: str = "keepvar"
    ) -> np.ndarray:
        if method not in ("in", "out", "sym", "keepvar"):
            raise ValueError("Unrecognized method!")
        vnum, i, j, ew, _ = subgraph
        subgraph_coo = sp.coo_array((ew, (i, j)), shape=(vnum, vnum))
        # _, i, j, _, enorm = tuple_net
        if method in ("in", "keepvar", "sym"):
            in_degrees = GraphDataLoader.vertex_degrees(
                subgraph_coo, direction="in"
            )
            in_normalizer = np.power(
                in_degrees[j], -1 if method == "in" else -0.5
            )
            # In case there are unconnected vertices
            in_normalizer[~np.isfinite(in_normalizer)] = 0
            ew = ew * in_normalizer
        if method in ("out", "sym"):
            out_degrees = GraphDataLoader.vertex_degrees(
                subgraph_coo, direction="out"
            )
            out_normalizer = np.power(
                out_degrees[i], -1 if method == "out" else -0.5
            )
            # In case there are unconnected vertices
            out_normalizer[~np.isfinite(out_normalizer)] = 0
            ew = ew * out_normalizer
        return ew

    @staticmethod
    def normalize_subgraph(subgraph: SUBGRAPH) -> NORMED_SUBGRAPH:
        vnum, i, j, ew, es = subgraph
        i = torch.tensor(i, dtype=torch.long)
        j = torch.tensor(j, dtype=torch.long)
        es = torch.tensor(es, dtype=torch.float32)
        ew = torch.tensor(ew, dtype=torch.float32)

        ew_norm = GraphDataLoader.normalize_edges(subgraph)
        ew_norm = torch.tensor(
            np.concatenate([ew_norm, np.zeros(len(i) - len(ew_norm))]),
            dtype=torch.float32,
        )

        ind = torch.randperm(i.size(0))
        return i[ind], j[ind], es[ind], ew[ind], ew_norm[ind]

    def __init__(
        self,
        mdata: MuData,
        net_key: str,
        batch_size: Union[float, int],
        drop_self_loop: bool = True,
        num_negative_samples: int = 1,
        phase: Literal["train", "test"] = "train",
    ) -> None:
        # copy graph, doesn't affect the original graph (use csr_array)
        net: sp.csr_array = sp.csr_array(mdata.varp[net_key])
        # check symmetry
        assert (net.T - net).nnz == 0, "mmAAVI just accept symmetric graph."
        assert phase in ["train", "test"], "phase must be train or test."

        # add self-loop
        if (net.diagonal() != 0).any():
            warnings.warn(
                "graph has non-zero diagonal elements, "
                "they will be set as 1. "
            )
        net.setdiag(1)

        self._phase = phase
        self._net_key = net_key
        self._dsl = drop_self_loop
        self._nns = num_negative_samples
        self._bs = batch_size
        self._net = net

        if isinstance(self._bs, float):
            assert (self._bs > 0.0) and (self._bs < 1.0)
            n_edges = self._net.nnz
            # negative sampling
            self._bs = int(n_edges * (1 + self._nns) * self._bs)
        logging.info("network batch size is %d" % self._bs)

        if self._phase == "test":
            net_coo = self._net.tocoo()
            if self._dsl:
                net_coo.setdiag(0)
            i, j, ew = net_coo.row, net_coo.col, net_coo.data
            es = 2 * (ew > 0) - 1
            ew = np.abs(ew)
            tupled_graph = (net_coo.shape[0], i, j, ew, es)
            self._normed_subgraph = self.normalize_subgraph(tupled_graph)

    def __iter__(self):
        if self._phase == "test":
            return iter([{"normed_subgraph": self._normed_subgraph}])

        subgraph = self.sample_subgraph(
            self._net, self._nns, drop_self_loop=self._dsl
        )
        self._normed_subgraph = self.normalize_subgraph(subgraph)
        # Calculate the total number of iterations.
        # Set a counter variable and stop the iteration when it exceeds the
        # total number of iterations.
        n_edge_subgraph = self._normed_subgraph[0].size(0)
        self._iter_index = 0
        self._iter_length = (n_edge_subgraph + self._bs - 1) // self._bs
        # self.n_sample_net = self.sampled_net[0].shape[0]
        # self.n = (self.n_sample_net + self.bs_int - 1) // self.bs_int

        return self

    def __next__(self) -> Dict[str, NORMED_SUBGRAPH]:
        if self._iter_index >= self._iter_length:
            raise StopIteration
        index = slice(
            self._iter_index * self._bs, (self._iter_index + 1) * self._bs
        )
        graph_minibatch = tuple(a[index] for a in self._normed_subgraph)
        self._iter_index += 1
        return {
            "graph_minibatch": graph_minibatch,
            "normed_subgraph": self._normed_subgraph,
        }


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
        cycle_flags: Optional[List[bool]] = None,
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

    def _next(self, i: int) -> GMINIBATCH:
        try:
            return next(self.iterators[i])
        except StopIteration as e:
            if self.cycle_flags[i]:
                self.iterators[i] = iter(self.data_loaders[i])
                return next(self.iterators[i])
            raise e

    def __next__(self) -> GMINIBATCH:
        res = {}
        for i in range(self.num_loaders):
            res.update(self._next(i))
        return res


def get_dataloader(
    mdata: MuData,
    input_key: Optional[str] = "log1p_norm",
    output_key: Optional[str] = None,
    batch_key: str = "batch",
    dlabel_key: Optional[str] = None,
    sslabel_key: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    net_key: Optional[str] = None,
    graph_batch_size: Union[float, int] = 0.5,
    drop_self_loop: bool = True,
    num_negative_samples: int = 1,
    graph_data_phase: Literal["train", "test"] = "train",
    resample_size: Optional[int] = None,
    balance_sample_size: Optional[Union[str, int]] = None,
    label_ratio: float = 0.2,
    repeat_sample: bool = True,
    drop_last: bool = False,
) -> Union[D.DataLoader, ParallelDataLoader]:
    """
    resample_size: The number of random resamples. If it is None, resampling
    is not performed. This is used in differential.

    balance_sample_size: Specifies the number of samples per batch to balance
    the sample sizes during training. If it is None, balance_sample_size
    is not performed.
    resample_size and balance_sample_size cannot be enabled simultaneously!!!

    graph_data_phase: If it is test, each batch only returns the
    complete normed graph.
    """

    assert (resample_size is None) or (
        balance_sample_size is None
    ), "resample_size and balance_sample_size can not be set at the same time."

    mdataset = MosaicMuDataset(
        mdata,
        input_key=input_key,
        output_key=output_key,
        batch_key=batch_key,
        dlabel_key=dlabel_key,
        sslabel_key=sslabel_key,
    )
    if sslabel_key is not None:
        # should use SemiBatchSampler
        sssampler = SemiSupervisedSampler(
            mdata,
            sslabel_key,
            batch_size=batch_size,
            label_ratio=label_ratio,
            shuffle=shuffle,
            repeat_sample=repeat_sample,
            # If balance_sample_size is None, balance sampling is not
            # performed. However, in SemiSupervisedSampler, whether to perform
            # balance sampling is controlled by balance_label_key.
            balance_label_key=(
                None if balance_sample_size is None else batch_key
            ),
            balance_sample_size=balance_sample_size,
        )
        mdataloader = D.DataLoader(
            mdataset,
            batch_sampler=sssampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif resample_size is None and balance_sample_size is None:
        mdataloader = D.DataLoader(
            mdataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    else:
        if resample_size is not None:
            sampler = D.RandomSampler(
                mdataset, replacement=True, num_samples=resample_size
            )
        elif balance_sample_size is not None:
            sampler = BalanceSizeSampler(mdata, batch_key, balance_sample_size)
        mdataloader = D.DataLoader(
            mdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    if net_key is None:
        return mdataloader

    graphloader = GraphDataLoader(
        mdata,
        net_key,
        graph_batch_size,
        drop_self_loop,
        num_negative_samples,
        phase=graph_data_phase,
    )
    return ParallelDataLoader(
        mdataloader, graphloader, cycle_flags=[False, True]
    )
