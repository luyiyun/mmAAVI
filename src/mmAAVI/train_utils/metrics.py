from math import log
from typing import Optional, Tuple, Union

# import seaborn as sns
import igraph as ig
import leidenalg
import numpy as np
# import cupy as cp
import pandas as pd
import torch
# from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse.csgraph import connected_components
# from scipy.special import binom
from scipy.stats import chi2, entropy
# from torch.utils.data import Subset
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# from cuml import KMeans, DBSCAN
# from cuml.manifold.umap import UMAP
# from sklearn.cluster import KMeans, DBSCAN
# from umap import UMAP
# from omegaconf import OmegaConf


# def umap_and_plot(z, clu, labels, batches):
#     if clu.ndim == 2:
#         clu = clu.argmax(dim=1)
#     # ------------------------------------------------------------------------
#     # 使用cuml
#     # cuml无法通过参数指定使用的devices，所以我们直接将pytorch使用的device也固定
#     #   为第一个gpu，然后通过环境变量CUDA_VISIBLE_DEVICES来指定训练时使用哪个gpu，
#     #   这样可以保证cuml和pytorch使用一个gpu
#     umap_op = UMAP(n_components=2, random_state=0)
#     x_umap = umap_op.fit_transform(z)
#     x_umap, clu = cp.asnumpy(x_umap), cp.asnumpy(clu)
#     # ------------------------------------------------------------------------
#     # 使用umap(cpu)
#     # z, clu = z.cpu().numpy(), clu.cpu().numpy()
#     # umap_op = UMAP(
#     #     n_components=2, n_neighbors=30, min_dist=0.2, random_state=0,
#     #     init="random"  # init使用random后会提高性能
#     # )
#     # x_umap = umap_op.fit_transform(z)
#     # ------------------------------------------------------------------------
#     x_umap = pd.DataFrame(x_umap, columns=["Z1", "Z2"])
#     for name, arr in zip(
#         ["batch", "label", "cluster"], [batches, labels, clu]
#     ):
#         x_umap[name] = arr
#         x_umap[name] = x_umap[name].astype("category")
#     fg_label = sns.relplot(
#         data=x_umap, x="Z1", y="Z2", hue="label", col="batch", col_wrap=2, s=5,
#         height=4
#     )
#     fg_cluster = sns.relplot(
#         data=x_umap, x="Z1", y="Z2", hue="cluster", col="batch", col_wrap=2,
#         s=5, height=4
#     )
#     return {"label": fg_label, "cluster": fg_cluster}
#
#
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
        # TODO: 还可以使用cuml进一步进行加速
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


############################################################################
#
# ARI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
############################################################################


# def ari(group1, group2, implementation=None):
#     """Adjusted Rand Index
#     The function is symmetric, so group1 and group2 can be switched
#     For single cell integration evaluation the scenario is:
#         predicted cluster assignments vs. ground-truth (e.g. cell type)
#         assignments
#     :param adata: anndata object
#     :param group1: string of column in adata.obs containing labels
#     :param group2: string of column in adata.obs containing labels
#     :params implementation: of set to 'sklearn', uses sklearns
#         implementation, otherwise native implementation is taken
#     """
#
#     if len(group1) != len(group2):
#         raise ValueError(
#             f"different lengths in group1 ({len(group1)}) "
#             f"and group2 ({len(group2)})"
#         )
#
#     if implementation == "sklearn":
#         return adjusted_rand_score(group1, group2)
#
#     def binom_sum(x, k=2):
#         return binom(x, k).sum()
#
#     n = len(group1)
#     contingency = pd.crosstab(group1, group2)
#
#     ai_sum = binom_sum(contingency.sum(axis=0))
#     bi_sum = binom_sum(contingency.sum(axis=1))
#
#     index = binom_sum(np.ravel(contingency))
#     expected_index = ai_sum * bi_sum / binom_sum(n, 2)
#     max_index = 0.5 * (ai_sum + bi_sum)
#
#     return (index - expected_index) / (max_index - expected_index)


#############################################################################
#
# NMI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
#############################################################################


# def nmi(group1, group2, method="arithmetic", nmi_dir=None):
#     """
#     Wrapper for normalized mutual information NMI between two different
#     cluster assignments
#     :param adata: Anndata object
#     :param group1: column name of `adata.obs`
#     :param group2: column name of `adata.obs`
#     :param method: NMI implementation
#         'max': scikit method with `average_method='max'`
#         'min': scikit method with `average_method='min'`
#         'geometric': scikit method with `average_method='geometric'`
#         'arithmetic': scikit method with `average_method='arithmetic'`
#         'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
#             https://sites.google.com/site/andrealancichinetti/mutual
#         'ONMI': implementation by Aaron F. McDaid et al.
#             https://github.com/aaronmcdaid/Overlapping-NMI
#     :param nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI'
#         are specified as `method`. These packages need to be compiled as
#         specified in the corresponding READMEs.
#     :return:
#         Normalized mutual information NMI value
#     """

#     if len(group1) != len(group2):
#         raise ValueError(
#             f'different lengths in group1 ({len(group1)}) '
#             f'and group2 ({len(group2)})'
#         )

#     # choose method
#     if method in ['max', 'min', 'geometric', 'arithmetic']:
#         nmi_value = normalized_mutual_info_score(group1, group2,
#                                                  average_method=method)
#     elif method == "Lancichinetti":
#         nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
#     elif method == "ONMI":
#         nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
#     else:
#         raise ValueError(f"Method {method} not valid")

#     return nmi_value


# def onmi(group1, group2, nmi_dir=None, verbose=True):
#     """
#     Based on implementation https://github.com/aaronmcdaid/Overlapping-NMI
#     publication: Aaron F. McDaid, Derek Greene, Neil Hurley 2011
#     params:
#         nmi_dir: directory of compiled C code
#     """

#     if nmi_dir is None:
#         raise FileNotFoundError(
#             "Please provide the directory of the compiled C code from "
#             "https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
#         )

#     group1_file = write_tmp_labels(group1, to_int=False)
#     group2_file = write_tmp_labels(group2, to_int=False)

#     nmi_call = subprocess.Popen(
#         [nmi_dir + "onmi", group1_file, group2_file],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)

#     stdout, stderr = nmi_call.communicate()
#     if stderr:
#         print(stderr)

#     nmi_out = stdout.decode()
#     if verbose:
#         print(nmi_out)

#     nmi_split = [x.strip().split('\t') for x in nmi_out.split('\n')]
#     nmi_max = float(nmi_split[0][1])

#     # remove temporary files
#     os.remove(group1_file)
#     os.remove(group2_file)

#     return nmi_max


# def nmi_Lanc(group1, group2, nmi_dir="external/mutual3/", verbose=True):
#     """
#     paper by A. Lancichinetti 2009
#     https://sites.google.com/site/andrealancichinetti/mutual
#     recommended by Malte
#     """

#     if nmi_dir is None:
#         raise FileNotFoundError(
#             "Please provide the directory of the compiled C code from "
#             "https://sites.google.com/site/andrealancichinetti/"
#             "mutual3.tar.gz"
#         )

#     group1_file = write_tmp_labels(group1, to_int=False)
#     group2_file = write_tmp_labels(group2, to_int=False)

#     nmi_call = subprocess.Popen(
#         [nmi_dir + "mutual", group1_file, group2_file],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)

#     stdout, stderr = nmi_call.communicate()
#     if stderr:
#         print(stderr)
#     nmi_out = stdout.decode().strip()

#     return float(nmi_out.split('\t')[1])


# def write_tmp_labels(group_assignments, to_int=False, delim='\n'):
#     """
#     write the values of a specific obs column into a temporary file in text
#     format needed for external C NMI implementations (onmi and nmi_Lanc
#     functions), because they require files as input
#     params:
#         to_int: rename the unique column entries by integers in
#             range(1,len(group_assignments)+1)
#     """
#     import tempfile

#     if to_int:
#         label_map = {}
#         i = 1
#         for label in set(group_assignments):
#             label_map[label] = i
#             i += 1
#         labels = delim.join([str(label_map[name])
#                              for name in group_assignments])
#     else:
#         labels = delim.join([str(name) for name in group_assignments])

#     clusters = {label: [] for label in set(group_assignments)}
#     for i, label in enumerate(group_assignments):
#         clusters[label].append(str(i))

#     output = '\n'.join([' '.join(c) for c in clusters.values()])
#     output = str.encode(output)

#     # write to file
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         f.write(output)
#         filename = f.name

#     return filename
