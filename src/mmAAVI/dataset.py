import logging
import warnings
from typing import (
    Set,
    List,
    Optional,
    Union,
    Dict,
    Tuple,
    TypedDict,
)

import numpy as np
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
        # TODO:
        if batch_key is None:
            raise NotImplementedError

        super().__init__()

        self._mdata = mdata
        self._input_key = input_key
        self._output_key = output_key
        self._batch_key = batch_key
        self._dlabel_key = dlabel_key if dlabel_key is not None else batch_key
        self._sslabel_key = sslabel_key
        # self._nrows_cum = np.cumsum(data.batch_dims)
        # self._imp_miss = impute_miss

        # prepare input and output ...
        # self._in_eq_out = (
        #     input_use is None and output_use is None
        # ) or input_use == output_use
        # self._inpt_grid = data.reps[input_use] if input_use else data.X
        # self._oupt_grid = data.reps[output_use] if output_use else data.X
        # self._inpt_grid.as_sparse_type("csr")
        # self._oupt_grid.as_sparse_type("csr")

        # # prepare meta variables ...
        # self._use_obs = pd.DataFrame(index=data.obs.index)
        # # 处理blabel和dlabel
        # bl = "_batch" if obs_blabel is None else obs_blabel
        # dl = "_batch" if obs_dlabel is None else obs_dlabel
        # self._blabel = data.obs[bl].astype("category")
        # self._blabel_codes = self._blabel.cat.codes.values
        # self._dlabel = data.obs[dl].astype("category")
        # self._dlabel_codes = self._dlabel.cat.codes.values
        # # 处理sslabel
        # if obs_sslabel is not None:
        #     sslabel = data.obs[obs_sslabel].copy()
        #     sslabel.replace(sslabel_mapping, inplace=True)
        #     sslabel.fillna(-1, inplace=True)
        #     self._sslabel_codes = sslabel.values
        #     # 记录sslabel的无缺失版本，用于计算指标
        #     if obs_sslabel_full is not None:
        #         sslabelf = data.obs[obs_sslabel_full].copy()
        #         sslabelf.replace(sslabel_mapping, inplace=True)
        #         self._sslabel_full_codes = sslabelf.values
        #     else:
        #         self._sslabel_full_codes = None
        # else:
        #     self._sslabel_codes = None
        #     self._sslabel_full_codes = None

        # self._keys_od = ["input", "output", "mask"]
        # if self._imp_miss:
        #     self._keys_od.append("imp_blabel")
        # self._keys_od = np.array(self._keys_od).astype(np.str_)

        # self._keys_arr = ["blabel", "dlabel"]
        # if self._sslabel_codes is not None:
        #     self._keys_arr.append("sslabel")
        # self._keys_arr = np.array(self._keys_arr).astype(np.str_)

    def __len__(self) -> int:
        return self._mdata.n_obs

    def __getitem__(self, index: int) -> SAMPLE:
        row = self._mdata[index, :]
        inpts, outputs, mask = {}, {}, {}
        for k, row_ann in row.mod.items():
            mask[k] = self._mdata.obsm[k][index]
            # NOTE: row_ann的n_obs为0并不能证明这个组学缺失，还有可能是因为这个组学
            # 的count就是0，正确的做法是通过mudata的obsm["OMIC_NAME"][i]来判断
            if row_ann.n_obs == 0:
                inpts[k] = torch.zeros(row_ann.n_vars, dtype=torch.float32)
                outputs[k] = torch.zeros(row_ann.n_vars, dtype=torch.float32)
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


# class PairBatchRandomSampler:
#     def __init__(
#         self,
#         blabel1: np.ndarray,
#         blabel2: np.ndarray,
#         nsamples: int,
#         dat_ind: int = 1,
#         seed: int = 0,
#     ) -> None:
#         assert dat_ind in [1, 2]

#         batch1_uni = np.unique(blabel1)
#         batch2_uni = np.unique(blabel2)
#         self.blabel_uni = np.intersect1d(batch1_uni, batch2_uni)

#         # preprocess batch indices
#         self.batch1_dict = {
#             k: np.nonzero(blabel1 == k)[0] for k in self.blabel_uni
#         }
#         self.batch2_dict = {
#             k: np.nonzero(blabel2 == k)[0] for k in self.blabel_uni
#         }

#         # calculate the probabilities of each batch
#         cnt = pd.value_counts(np.concatenate([blabel1, blabel2]))
#         cnt = cnt.loc[self.blabel_uni]
#         cnt /= cnt.sum()

#         # calculate the
#         self.bratio = cnt.values
#         self.nsamples = nsamples
#         self.dat_ind = dat_ind
#         self.rng = np.random.default_rng(seed)

#     def __iter__(self):
#         bsamples = self.rng.multinomial(self.nsamples, self.bratio)
#         blabel_dict = (
#             self.batch1_dict if self.dat_ind == 1 else self.batch2_dict
#         )
#         res = []
#         for k, bni in zip(self.blabel_uni, bsamples):
#             resi = self.rng.choice(blabel_dict[k], bni)
#             res.append(resi)
#         res = np.concatenate(res)
#         return iter(res)

#     def __len__(self):
#         return self.nsamples


class GraphDataLoader:  # TODO: valid phase时不用sample graph
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
        从graph中进行负采样，其中采样的edge的起点已经确定（i_neg）， 但是终点还没有。
        采样过程中需要保证终点被采样的概率服从某个给定的概率
        graph只会用到其起点和终点坐标，边的值不会用到，所以不需要进行abs和binary
        该实现可能会造成infinity loop
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
        该实现是循环地针对每个起点进行采样，采样中只针对其非邻节点采样
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
        vnum = net.shape[0]
        net_coo = net.tocoo()
        i, j, ew = net_coo.row, net_coo.col, net_coo.data
        es = 2 * (ew > 0) - 1
        ew = np.abs(ew)
        eset = set(zip(i, j))
        # net_abs = abs(net)
        # net_abs_coo = net_abs.tocoo()
        # i, j, ew = net_abs_coo.row, net_abs_coo.col, net_abs_coo.data

        # shape, i, j, es, ew = tuple_net
        # assert shape[0] == shape[1], "just suitable symmetric matrix"
        # vnum = shape[0]
        # # 去掉self-loop
        # if drop_self_loop:
        #     ind = i != j
        #     i, j, es, ew = i[ind], j[ind], es[ind], ew[ind]
        # TODO: 这里和我原来写的有点区别。这里setdiag会影响到degree的计算；但是原来
        # 的写法是不会影响到degree的计算的
        if drop_self_loop:
            net.setdiag(0)

        # == calculate probability of vertices
        # eset = set(zip(i, j))
        # if weighted_sampling:
        #     degree = vertex_degrees(tuple_net, direction="both")
        # else:
        #     degree = np.ones(vnum, dtype=ew.dtype)
        degree = (
            GraphDataLoader.vertex_degrees(abs(net))
            if weighted_sampling
            else np.ones(vnum, dtype=float)
        )
        # NOTE: == 下面的code并未改动 ==
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
        # pi_, pj_, pw_, ps_ = i[psamp], j[psamp], ew[psamp], es[psamp]
        # pw_ = np.ones_like(pw_)

        # == sample the negative edges, prefer bigger vertex prob
        i_neg = np.tile(i_pos, neg_samples)
        ew_neg = np.zeros(effective_enum * neg_samples, dtype=np.float32)
        es_neg = np.tile(es_pos, neg_samples)
        j_neg = GraphDataLoader.negative_sample_from_graph_1(
            vnum, eset, i_neg, vprob
        )
        # nw_ = np.zeros(pw_.size * neg_samples, dtype=pw_.dtype)
        # ns_ = np.tile(ps_, neg_samples)
        # nj_ = np.random.choice(
        #     vnum, pj_.size * neg_samples, replace=True, p=vprob
        # )
        # remain = np.where([item in eset for item in zip(ni_, nj_)])[0]
        # while remain.size:  #  Potential infinite loop if graph too dense
        #     newnj = np.random.choice(vnum, remain.size,
        #                              replace=True, p=vprob)
        #     nj_[remain] = newnj
        #     remain = remain[[item in eset for item in
        #               zip(ni_[remain], newnj)]]

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
        # net_style: Literal["dict", "sarr", "walk"],
        batch_size: Union[float, int],
        # num_workers: Optional[int] = 0,
        drop_self_loop: bool = True,
        # walk_length: int = 20,
        # context_size: Optional[int] = 10,
        # walks_per_node: int = 10,
        num_negative_samples: int = 1,
        # p: float = 1.0,
        # q: float = 1.0,
    ) -> None:
        # copy graph, doesn't affect the original graph (use csr_array)
        net: sp.csr_array = sp.csr_array(mdata.varp[net_key])
        # check symmetry
        assert (net.T - net).nnz == 0, "mmAAVI just accept symmetric graph."

        # add self-loop
        if (net.diagonal() != 0).any():
            warnings.warn(
                "graph has non-zero diagonal elements, "
                "they will be set as 1. "
            )
        net.setdiag(1)

        # if net_style != "sarr":
        #     raise NotImplementedError
        # assert batch_size is not None
        # assert nets[0][0] == nets[0][1], "the dims of network must be equal"

        self._net_key = net_key
        self._dsl = drop_self_loop
        self._nns = num_negative_samples
        self._bs = batch_size
        self._net = net
        # self._np_generator = np.random.default_rng(seed)
        # self._th_generator = torch.Generator()

        # self._index = None

        # self.nets = nets
        # self.net_style = net_style
        # self.n = np.inf  # 如果没有设置n，则n是无穷大
        # self.bs = batch_size
        # self.nw = num_workers
        # self.drop_self_loop = drop_self_loop
        # self.vnum = nets[0][0]
        # self.nns = num_negative_samples
        # logging.info("number of edges in network is %d" % (nets[1].shape[0]))

        if isinstance(self._bs, float):
            assert (self._bs > 0.0) and (self._bs < 1.0)
            # if self.net_style == "walk":
            #     self.bs_int = int(self.bs * self.vnum)
            # elif self.net_style == "sarr":
            # if self._dsl:
            #     n_edges = (self.nets[1] != self.nets[2]).sum()
            # else:
            #     n_edges = self.nets[1].shape[0]
            n_edges = self._net.nnz
            self._bs = int(n_edges * (1 + self._nns) * self._bs)  # 负采样
        # else:
        #     self.bs_int = self.bs
        logging.info("network batch size is %d" % self._bs)

        # if self.net_style == "walk":
        #     pass
        #     self.walk_length = walk_length
        #     self.context_size = context_size
        #     self.walks_per_node = walks_per_node
        #     self.num_negative_samples = num_negative_samples
        #     self.p = p
        #     self.q = q
        #     self.random_walk_fn = torch.ops.torch_cluster.random_walk

        #     if self.context_size is not None:
        #         self.num_walks_per_rw = (
        #             1 + self.walk_length + 1 - self.context_size
        #         )

        #     edge_index = torch.tensor(
        #         np.stack([nets[1], nets[2]], axis=0), dtype=torch.long
        #     )
        #     row, col = sort_edge_index(edge_index, num_nodes=self.vnum).cpu()
        #     self.walk_need = index2ptr(row, self.vnum), col
        #     self.net_sign = torch.tensor(self.nets[3]).float()

        #     self.walk_loader = D.DataLoader(
        #         range(self.vnum),
        #         batch_size=self.bs_int,
        #         shuffle=True,
        #         num_workers=self.nw,
        #         collate_fn=self.sample_rw,
        #     )

    def __iter__(self):
        subgraph = self.sample_subgraph(
            self._net, self._nns, drop_self_loop=self._dsl
        )
        self._normed_subgraph = self.normalize_subgraph(subgraph)
        # 计算总的迭代数量，设置一个计数变量，当其超过迭代总量时停止迭代
        n_edge_subgraph = self._normed_subgraph[0].size(0)
        self._iter_index = 0
        self._iter_length = (n_edge_subgraph + self._bs - 1) // self._bs
        # self.n_sample_net = self.sampled_net[0].shape[0]
        # self.n = (self.n_sample_net + self.bs_int - 1) // self.bs_int

        # if self.net_style == "walk":
        #     # def generator():
        #     #     for walk in self.sampled_loader:
        #     #         yield {"walk": walk, "coo": self.sarr_sign}
        #     # return generator()
        #     return iter(self.walk_loader)
        # if self.net_style == "sarr":
        #     self.sample_subgraph()
        # self._index = 0
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

        # if self.net_style == "dict":
        #     net = {}
        #     for k, sarr in self.nets.items():
        #         negi, negj = sample_negs1(sarr, sarr.nnz)
        #         posi, posj, wt = sarr.row, sarr.col, sarr.data
        #         esign = (wt > 0).astype(float)
        #         wt = np.abs(wt)

        #         iall = torch.tensor(
        #             np.concatenate([posi, negi]), dtype=torch.long
        #         )
        #         jall = torch.tensor(
        #             np.concatenate([posj, negj]), dtype=torch.long
        #         )
        #         sall = torch.tensor(
        #             np.concatenate([esign, np.ones_like(esign)]),
        #             dtype=torch.float32,
        #         )
        #         wall = torch.tensor(
        #             np.concatenate([wt, np.zeros_like(wt)]),
        #             dtype=torch.float32,
        #         )
        #         ind = torch.randperm(iall.size(0))
        #         net[k] = (iall[ind], jall[ind], sall[ind], wall[ind])
        #     return {"varp": net}
        # elif self.net_style == "sarr":
        #     index = slice(
        #         self._index * self.bs_int, (self._index + 1) * self.bs_int
        #     )
        #     sample_graph = tuple(a[index] for a in self.sampled_net)
        #     return {"sample_graph": sample_graph, "varp": self.sampled_net}
        # elif self.net_style == "walk":
        #     raise NotImplementedError  # walk使用其他的迭代器

        # self._index += 1

    # def sample_graph(self) -> None:
    #     iall, jall, sall, wall = sample_negs2(
    #         self.nets, neg_samples=self.nns,
    #         drop_self_loop=self.drop_self_loop
    #     )
    #     enorm = normalize_edges(self.nets)
    #     enorm_all = torch.tensor(
    #         np.concatenate([enorm, np.zeros(len(iall) - len(enorm))]),
    #         dtype=torch.float32,
    #     )
    #     iall = torch.tensor(iall, dtype=torch.long)
    #     jall = torch.tensor(jall, dtype=torch.long)
    #     sall = torch.tensor(sall, dtype=torch.float32)
    #     wall = torch.tensor(wall, dtype=torch.float32)
    #     ind = torch.randperm(iall.size(0))
    #     self.sampled_net = (
    #         iall[ind],
    #         jall[ind],
    #         sall[ind],
    #         wall[ind],
    #         enorm_all[ind],
    #     )
    #     self.n_sample_net = self.sampled_net[0].shape[0]
    #     self.n = (self.n_sample_net + self.bs_int - 1) // self.bs_int

    # def sample_rw(self, batch: T):
    #     if not isinstance(batch, torch.Tensor):
    #         batch = torch.tensor(batch)

    #     # pos_sample
    #     start = batch.repeat(self.walks_per_node)
    #     pos_rw, ed_ind = self.random_walk_fn(
    #         *self.walk_need,
    #         start,
    #         self.walk_length,
    #         self.p,
    #         self.q,
    #     )
    #     pos_sign = self.net_sign[ed_ind]

    #     # neg_sample
    #     start = batch.repeat(self.walks_per_node * self.num_negative_samples)
    #     neg_rw = torch.randint(
    #         self.vnum,
    #         (start.size(0), self.walk_length),
    #         dtype=start.dtype,
    #         device=start.device,
    #     )
    #     neg_rw = torch.cat([start.view(-1, 1), neg_rw], dim=-1)

    #     if self.context_size is not None:
    #         pos_rw_s, neg_rw_s, pos_sign_s = [], [], []
    #         for j in range(self.num_walks_per_rw):
    #             pos_rw_s.append(pos_rw[:, j : (j + self.context_size)])
    #             neg_rw_s.append(neg_rw[:, j : (j + self.context_size)])
    #             pos_sign_s.append(pos_sign[:, j:(j + self.context_size - 1)])
    #         pos_rw = torch.cat(pos_rw_s, dim=0)
    #         neg_rw = torch.cat(neg_rw_s, dim=0)
    #         pos_sign = torch.cat(pos_sign_s, dim=0)

    #     return {"walk": (pos_rw, neg_rw), "pos_sign": pos_sign}


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


# class BalanceSizeSampler:
#     def __init__(
#         self,
#         resample_size: dict[str, float],
#         blabels: np.ndarray,
#     ) -> None:
#         self._resample_size = resample_size
#         self._batch_index_dict, self._n = {}, 0
#         blabel_uni = np.unique(blabels)
#         for bi in blabel_uni:
#             ind = np.nonzero(blabels == bi)[0]
#             self._batch_index_dict[bi] = ind
#             self._n += int(ind.shape[0] * self._resample_size.get(bi, 1.0))

#     def _get_index(self):
#         index = []
#         for bk, bi in self._batch_index_dict.items():
#             ratio = self._resample_size.get(bk, 1.0)
#             if ratio == 1.0:
#                 index.append(bi)
#             else:
#                 target_size = int(ratio * bi.shape[0])
#                 index.append(np.random.choice(bi, target_size, replace=True))
#         index = np.concatenate(index)
#         np.random.shuffle(index)
#         return index

#     def __iter__(self):
#         return iter(self._get_index())

#     def __len__(self) -> int:
#         return self._n


# class SemiSupervisedSampler:

#     """
#     为半监督生成batch，
#         保证每个batch都包含足够的unlabel和labeled样本。
#         他们两者是分别采样，然后放在一起的。
#         其总批次数量=批次数量较大的那个

#     需要同时控制random和np.random的随机性
#     """

#     def __init__(
#         self,
#         is_labeled: np.ndarray,
#         unlabel_batch_size: int,
#         labeled_batch_size: int,
#         resample_size: Optional[dict[str, float]] = None,
#         blabels: Optional[np.ndarray] = None,
#         drop_last: bool = True,
#     ) -> None:
#         self._is_labeled = is_labeled
#         self._ubs = unlabel_batch_size
#         self._lbs = labeled_batch_size
#         self._flag_resample = resample_size is not None
#         self._dl = drop_last

#         if self._flag_resample:
#             assert blabels is not None
#             self._balance_sampler = BalanceSizeSampler(resample_size,
#                                                        blabels)
#             self.n = len(self._balance_sampler)
#             # TODO: 为了让第一个epoch也有length，但是可能不准
#             index = self._balance_sampler._get_index()
#             is_labeled_re = self._is_labeled[index]
#             n_l = is_labeled_re.sum()
#             n_u = self.n - n_l
#             if self._dl:
#                 self.nb_l = n_l // self._lbs
#                 self.nb_u = n_u // self._ubs
#             else:
#                 self.nb_l = (n_l + self._lbs - 1) // self._lbs
#                 self.nb_u = (n_u + self._ubs - 1) // self._ubs
#             self.nb = max(self.nb_l, self.nb_u)
#         else:
#             self.n = self._is_labeled.shape[0]
#             self.n_l = self._is_labeled.sum()
#             self.n_u = self.n - self.n_l
#             if drop_last:
#                 self.nb_l = self.n_l // self._lbs
#                 self.nb_u = self.n_u // self._ubs
#             else:
#                 self.nb_l = (self.n_l + self._lbs - 1) // self._lbs
#                 self.nb_u = (self.n_u + self._ubs - 1) // self._ubs
#             self.nb = max(self.nb_l, self.nb_u)

#             self.inds_l = np.nonzero(self._is_labeled)[0]
#             self.inds_u = np.nonzero(np.logical_not(self._is_labeled))[0]

#     def _iter_wo_resample(self):
#         inds_l = self.inds_l.copy()
#         inds_u = self.inds_u.copy()

#         for i in range(self.nb):
#             li, ui = i % self.nb_l, i % self.nb_u
#             if li == 0:
#                 np.random.shuffle(inds_l)
#             if ui == 0:
#                 np.random.shuffle(inds_u)

#             batch_l = inds_l[(li * self._lbs) : ((li + 1) * self._lbs)]
#             batch_u = inds_u[(ui * self._ubs) : ((ui + 1) * self._ubs)]

#             yield np.r_[batch_l, batch_u]

#     def _iter_w_resample(self):
#         index = self._balance_sampler._get_index()
#         is_labeled_re = self._is_labeled[index]
#         n_l = is_labeled_re.sum()
#         n_u = self.n - n_l
#         if self._dl:
#             self.nb_l = n_l // self._lbs
#             self.nb_u = n_u // self._ubs
#         else:
#             self.nb_l = (n_l + self._lbs - 1) // self._lbs
#             self.nb_u = (n_u + self._ubs - 1) // self._ubs
#         self.nb = max(self.nb_l, self.nb_u)

#         self.inds_l = np.nonzero(is_labeled_re)[0]
#         self.inds_u = np.nonzero(np.logical_not(is_labeled_re))[0]

#         for batch in self._iter_wo_resample():
#             yield [index[i] for i in batch]

#     def __iter__(self):
#         if self._flag_resample:
#             return self._iter_w_resample()
#         else:
#             return self._iter_wo_resample()

#     def __len__(self):
#         return self.nb


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
) -> Union[D.DataLoader, ParallelDataLoader]:
    mdataset = MosaicMuDataset(
        mdata,
        input_key=input_key,
        output_key=output_key,
        batch_key=batch_key,
        dlabel_key=dlabel_key,
        sslabel_key=sslabel_key,
    )
    mdataloader = D.DataLoader(
        mdataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if net_key is None:
        return mdataloader

    graphloader = GraphDataLoader(
        mdata, net_key, graph_batch_size, drop_self_loop, num_negative_samples
    )
    return ParallelDataLoader(
        mdataloader, graphloader, cycle_flags=[False, True]
    )
