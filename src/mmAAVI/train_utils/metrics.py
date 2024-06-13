from math import log
from typing import Optional, Tuple, Union

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2, entropy
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def entropy_for_distant_labels(labels, nclu=None, norm=False):
    if norm:
        assert nclu is not None, "please give clear ncluster for norm."
    if torch.is_tensor(labels):
        _, cnts = torch.unique(labels, return_counts=True)
        p = cnts / labels.size(0)
        entr = -(p * p.log()).sum()
        entr = entr.item()
    elif labels.__class__.__name__ == "ndarray":
        _, cnts = np.unique(labels, return_counts=True)
        p = cnts / labels.shape[0]
        entr = -(p * np.log(p)).sum()
    else:
        raise TypeError(type(labels))
    if norm:
        entr = entr / np.log(nclu)
    return entr


def entropy_for_predict_proba(proba, norm=False):
    if torch.is_tensor(proba):
        logp = proba.log()
        min_real = torch.finfo(logp.dtype).min
        logp = torch.clamp(logp, min=min_real)
        p_log_p = proba * logp
        entr = -p_log_p.sum(dim=1)
    elif proba.__class__.__name__ == "ndarray":
        entr = entropy(proba, axis=1)
    else:
        raise TypeError(type(proba))
    if norm:
        entr = entr / log(proba.shape[1])
    return entr


def entropy_for_predict_logit(logit, norm=False):
    logp = torch.log_softmax(logit, dim=1)
    min_real = torch.finfo(logp.dtype).min
    logp = torch.clamp(logp, min=min_real)
    p_log_p = logp * logp.exp()
    entr = -p_log_p.sum(dim=1)
    if norm:
        entr = entr / log(logit.size(1))
    return entr


##############################################################################
#
# graph connectivity score from scIB
#   https://github.com/theislab/scib/tree/main/scib
# It measures the mixing of the batches, and assesses whether the kNN graph
# representation, ​G,​ of the integrated data directly connects all cells with
# the same cell identity label. It first construct a KNN graph on the
# integrated latent embedding, then it calculate that for cells of each cell
# type, how well the subgraph is connected.
# -------------------------------------------------
##############################################################################


def graph_connectivity(X=None, G=None, groups=None, k=10, n_jobs=0):
    """ "
    Quantify how connected the subgraph corresponding to each batch cluster is.
    Calculate per label: #cells_in_largest_connected_component/#all_cells
    Final score: Average over labels

    :param adata: adata with computed neighborhood graph
    :param label_key: name in adata.obs containing the cell identity labels
    """
    clust_res = []
    if X is not None:
        # calculate the adjacency matrix of the neighborhood graph
        G = kneighbors_graph(
            X,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
            n_jobs=n_jobs,
        )
    elif G is None:
        raise ValueError("Either X or G should be provided")

    # make sure all cells have labels
    assert groups.shape[0] == G.shape[0]

    for group in np.sort(np.unique(groups)):
        G_sub = G[groups == group, :][:, groups == group]
        _, labels = connected_components(
            csr_matrix(G_sub), connection="strong"
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)


##############################################################################
#
# kBET
# -------------------------------------------------
##############################################################################


def kBET(
    batch: np.ndarray,
    data: Optional[np.ndarray] = None,
    G: Optional[Union[csr_matrix, csr_array]] = None,
    K: Union[int, float] = 25,
    alpha: float = 0.05,
    n_jobs: int = 0,
    # random_state: int = 0,
    # temp_folder: str = None,
    # use_cache: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    The kBET metric is defined in [Büttner18]_, which measures if cells from
    different samples mix well in their local neighborhood.
    """
    assert data is not None or G is not None
    if isinstance(K, float):
        assert K > 0.0 and K < 1.0
        K = int(data.shape[0] * K)

    if data is None:
        nsamples = max(G.shape)

    if G is None:
        assert data.shape[0] == batch.shape[0]
        nsamples = data.shape[0]

    gdist = pd.value_counts(batch).values
    cnt_expect = gdist / nsamples * K
    nbatches = gdist.shape[0]

    if G is None:
        knn = NearestNeighbors(
            n_neighbors=K - 1, n_jobs=n_jobs
        )  # eliminate the first neighbor, which is the node itself
        knn.fit(data)
        _, indices = knn.kneighbors()
    if data is None:
        indices, indptr = G.indices, G.indptr
        indices = np.array(
            [indices[indptr[i]:indptr[i + 1]] for i in range(nsamples)]
        )
    knn_indices = np.concatenate(
        (np.arange(nsamples).reshape(-1, 1), indices), axis=1
    )  # add query as 1-nn

    eye = np.eye(nbatches)
    batch_oh = eye[batch, :]
    batch_knn = batch_oh[knn_indices]
    batch_cnt_knn = batch_knn.sum(axis=1)
    stats = ((batch_cnt_knn - cnt_expect) ** 2 / cnt_expect).sum(axis=-1)
    pvals = 1 - chi2.cdf(stats, nbatches - 1)

    accept_rate = (pvals >= alpha).mean()

    return accept_rate, stats, pvals


##############################################################################
#
# Leiden Clustering
# https://github.com/scverse/scanpy/blob/
# 2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/tools/_leiden.py
# -------------------------------------------------
##############################################################################


def leiden_cluster(
    X=None,
    G=None,
    n_neighbors=30,
    resolution=1,
    random_state=0,
    n_iterations=-1,
    directed=True,
    add_weights=False,
    n_jobs=0,
    **partition_kwargs,
):
    assert X is not None or G is not None
    partition_kwargs = dict(partition_kwargs)

    if G is None:
        G = kneighbors_graph(X, n_neighbors, n_jobs=n_jobs)
    # sparse matrix -> igraph
    G = G.tocoo()
    igg = ig.Graph(directed=directed)
    igg.add_vertices(max(G.shape))  # this adds adjacency.shape[0] vertices
    igg.add_edges(list(zip(G.row, G.col)))
    if add_weights:
        igg.es["weight"] = G.data

    partition_type = leidenalg.RBConfigurationVertexPartition
    partition_kwargs["n_iterations"] = n_iterations
    partition_kwargs["seed"] = random_state
    if resolution is not None:
        partition_kwargs["resolution_parameter"] = resolution
    part = leidenalg.find_partition(igg, partition_type, **partition_kwargs)
    return np.array(part.membership)
