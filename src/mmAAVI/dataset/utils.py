"""
some codes refers from
https://github.com/gao-lab/GLUE/blob/master/scglue/models/data.py
"""


import numpy as np
import scipy.stats as sst
from h5py import Group
from scipy import sparse as sp

from .. import typehint as typ


def coo2tuples(sarr: sp.coo_array) -> typ.TUPLE_NET:
    i, j, ew = sarr.row, sarr.col, sarr.data
    es = (ew > 0).astype(float)
    return sarr.shape, i, j, es, np.abs(ew)


def vertex_degrees(
    tuple_net: typ.TUPLE_NET, direction: str = "both"
) -> np.ndarray:
    r"""
    Compute vertex degrees

    Parameters
    ----------
    direction
        Direction of vertex degree, should be one of {"in", "out", "both"}

    Returns
    -------
    degrees
        Vertex degrees
    """
    shape, i, j, _, ew = tuple_net
    adj = sp.coo_array((ew, (i, j)), shape=shape)
    if direction == "in":
        return adj.sum(axis=0)
    elif direction == "out":
        return adj.sum(axis=1)
    elif direction == "both":
        return adj.sum(axis=0) + adj.sum(axis=1) - adj.diagonal()
    raise ValueError("Unrecognized direction!")


def normalize_edges(
    tuple_net: typ.TUPLE_NET, method: str = "keepvar"
) -> np.ndarray:
    r"""
    Normalize graph edge weights

    Parameters
    ----------
    method
        Normalization method, should be one of {"in", "out", "sym", "keepvar"}

    Returns
    -------
    enorm
        Normalized weight of edges (:math:`n_{edges}`)
    """
    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    _, i, j, _, enorm = tuple_net
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(tuple_net, direction="in")
        in_normalizer = np.power(in_degrees[j], -1 if method == "in" else -0.5)
        # In case there are unconnected vertices
        in_normalizer[~np.isfinite(in_normalizer)] = 0
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(tuple_net, direction="out")
        out_normalizer = np.power(
            out_degrees[i], -1 if method == "out" else -0.5
        )
        # In case there are unconnected vertices
        out_normalizer[~np.isfinite(out_normalizer)] = 0
        enorm = enorm * out_normalizer
    return enorm


def sample_negs1(
    X: sp.coo_array, n: int = 3, replace: bool = False
) -> np.ndarray:
    N = np.prod(X.shape)
    m = N - X.size  # number of zero elements
    if n == 0:
        return (np.array([]), np.array([]))
    elif (n < 0) or (not replace and m < n) or (replace and m == 0):
        raise ValueError(
            "{n} samples from {m} locations do not exist".format(n=n, m=m)
        )
    elif n / m > 0.5:
        # Y (in the else clause, below) would be pretty dense
        # so there would be no point
        # trying to use sparse techniques. So let's use hpaulj's idea
        # (https://stackoverflow.com/a/53577267/190597) instead.
        import warnings

        warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

        Y = sp.coo_array(X == 0)
        rows = Y.row
        cols = Y.col
        idx = np.random.choice(len(rows), size=n, replace=replace)
        return (rows[idx], cols[idx])
    else:
        X_row, X_col = X.row, X.col
        X_data = np.ones(X.size)
        X = sp.coo_array((X_data, (X_row, X_col)), shape=X.shape)

        h, w = X.shape
        Y = sp.coo_array(X.shape)
        Y_size = 0
        while Y_size < n:
            m = n - Y.size
            Y_data = np.concatenate([Y.data, np.ones(m)])
            Y_row = np.concatenate([Y.row, np.random.choice(h, size=m)])
            Y_col = np.concatenate([Y.col, np.random.choice(w, size=m)])
            Y = sp.coo_array((Y_data, (Y_row, Y_col)), shape=X.shape)
            # Remove values in Y where X is nonzero
            # This also consolidates (row, col) duplicates
            Y = sp.coo_array(Y - X.multiply(Y))
            if replace:
                Y_size = Y.data.sum()
            else:
                Y_size = Y.size
        if replace:
            rows = np.repeat(Y.row, Y.data.astype(int))
            cols = np.repeat(Y.col, Y.data.astype(int))
            idx = np.random.choice(rows.size, size=n, replace=False)
            result = (rows[idx], cols[idx])
        else:
            rows = Y.row
            cols = Y.col
            idx = np.random.choice(rows.size, size=n, replace=False)
            result = (rows[idx], cols[idx])
    return result


def sample_negs2(
    tuple_net: typ.TUPLE_NET,
    neg_samples: int = 1,
    weighted_sampling: bool = True,
    drop_self_loop: bool = True,
) -> tuple[np.ndarray, ...]:
    shape, i, j, es, ew = tuple_net
    assert shape[0] == shape[1], "just suitable symmetric matrix"
    vnum = shape[0]
    # 去掉self-loop
    if drop_self_loop:
        ind = i != j
        i, j, es, ew = i[ind], j[ind], es[ind], ew[ind]

    eset = set(zip(i, j))
    if weighted_sampling:
        degree = vertex_degrees(tuple_net, direction="both")
    else:
        degree = np.ones(vnum, dtype=ew.dtype)
    degree_sum = degree.sum()
    if degree_sum:
        vprob = degree / degree_sum  # Vertex sampling probability
    else:  # Possible when `deemphasize_loops` is set on a loop-only graph
        vprob = np.ones(vnum, dtype=ew.dtype) / vnum

    effective_enum = ew.sum()
    eprob = ew / effective_enum  # Edge sampling probability
    effective_enum = round(effective_enum)

    psamp = np.random.choice(ew.size, effective_enum, replace=True, p=eprob)
    pi_, pj_, pw_, ps_ = i[psamp], j[psamp], ew[psamp], es[psamp]
    pw_ = np.ones_like(pw_)
    ni_ = np.tile(pi_, neg_samples)
    nw_ = np.zeros(pw_.size * neg_samples, dtype=pw_.dtype)
    ns_ = np.tile(ps_, neg_samples)
    nj_ = np.random.choice(vnum, pj_.size * neg_samples, replace=True, p=vprob)

    remain = np.where([item in eset for item in zip(ni_, nj_)])[0]
    while remain.size:  # NOTE: Potential infinite loop if graph too dense
        newnj = np.random.choice(vnum, remain.size, replace=True, p=vprob)
        nj_[remain] = newnj
        remain = remain[[item in eset for item in zip(ni_[remain], newnj)]]
    iall = np.concatenate([pi_, ni_])
    jall = np.concatenate([pj_, nj_])
    wall = np.concatenate([pw_, nw_])
    sall = np.concatenate([ps_, ns_])

    # perm = np.random.permutation(iall.shape[0])
    # return iall[perm], jall[perm], sall[perm], wall[perm]
    return iall, jall, sall, wall


def convert_sparse_index(ind):
    return [ind] if isinstance(ind, (np.integer, int)) else ind


def handle_sparse_slice(sarr):
    if sarr.shape[0] == 1:
        return sarr.todense()[0, :]
    elif sarr.shape[1] == 1:
        return sarr.todense()
    else:
        return sarr


def is_slice_none(s: slice):
    return (
        isinstance(s, slice)
        and s.start is None
        and s.stop is None
        and s.step is None
    )


def get_values_from_h5_kwargs(
    name: str, h5obj: Group, kwargs: dict, dtype: str = "may_none"
):
    value = kwargs[name] if name in kwargs else h5obj.attrs.get(name, None)
    if dtype == "may_none":
        if value == 0:  # 在h5中不能储存None，所以以0替代
            value = None
    elif dtype == "bool":
        value = bool(value)

    return value


def save_sparse_in_h5(h5obj: Group, name: str, sparr: typ.SPARSE) -> Group:
    sp_type = sparr.__class__.__name__

    g = h5obj.create_group(name)
    g.attrs["shape"] = tuple(sparr.shape)
    g.attrs["type"] = sp_type
    g.create_dataset("data", data=sparr.data)
    if sp_type.startswith("csr") or sp_type.startswith("csc"):
        g.create_dataset("indices", data=sparr.indices)
        g.create_dataset("indptr", data=sparr.indptr)
    elif sp_type.startswith("coo"):
        g.create_dataset("row", data=sparr.row)
        g.create_dataset("col", data=sparr.col)
    else:
        raise NotImplementedError(sparr.__class__.__name__)

    return g


def read_sparse_in_h5(sparse_obj: Group) -> typ.SPARSE:
    sp_type = sparse_obj.attrs["type"]
    shape = sparse_obj.attrs["shape"]
    if sp_type.startswith("csr"):
        sparr = sp.csr_array(
            (
                sparse_obj["data"][:],
                sparse_obj["indices"][:],
                sparse_obj["indptr"][:],
            ),
            shape=shape,
        )
    elif sp_type.startswith("csc"):
        sparr = sp.csc_array(
            (
                sparse_obj["data"][:],
                sparse_obj["indices"][:],
                sparse_obj["indptr"][:],
            ),
            shape=shape,
        )
    elif sp_type.startswith("coo"):
        sparr = sp.coo_array(
            (
                sparse_obj["data"][:],
                (sparse_obj["row"][:], sparse_obj["col"][:]),
            ),
            shape=shape,
        )
    else:
        raise NotImplementedError(sp_type)

    return sparr


def quantile_norm(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of M genes
    by N samples, quantile normalization ensures all samples have the same
    spread of data (by construction).

    The data across each row are averaged to obtain an average column. Each
    column quantile is replaced with the corresponding quantile of the average
    column.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with M rows (genes/features) and N columns (samples).

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # compute the quantiles
    quantiles = np.mean(np.sort(X, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 1, the
    # second-smallest by 2, ..., and the largest by M, the number of rows.
    ranks = np.apply_along_axis(sst.rankdata, 0, X)

    # convert ranks to integer indices from 0 to M-1
    rank_indices = ranks.astype(int) - 1

    # index the quantiles for each rank with the ranks matrix
    Xn = quantiles[rank_indices]

    return Xn


def quantile_norm_log(X, log=True):
    if log:
        logX = np.log1p(X)
    else:
        logX = X
    logXn = quantile_norm(logX)
    return logXn


def preprocess(counts, modality="RNA", log=True):
    if modality == "ATAC":
        # make binary, maximum is 1
        counts = (counts > 0).astype(np.float)
        # # normalize according to library size
        # counts = counts / np.sum(counts, axis = 1)[:,None]
        # counts = counts/np.max(counts)

    elif modality == "interaction":
        # gene by region matrix
        counts = counts / (np.sum(counts, axis=1)[:, None] + 1e-6)

    else:
        # other cases, e.g. Protein, RNA, etc
        counts = quantile_norm_log(counts, log=log)
        counts = counts / np.max(counts)

    return counts
