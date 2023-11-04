from collections import OrderedDict

import h5py
import pandas as pd
import scipy.sparse as sp


def read_sparse_in_h5(sparse_obj, return_matrix=True):
    sp_type = sparse_obj.attrs["type"]
    shape = sparse_obj.attrs["shape"]
    # 使用matrix，不然r读不到
    if sp_type.startswith("csr"):
        cls_func = sp.csr_matrix if return_matrix else sp.csr_array
        sparr = cls_func((
            sparse_obj["data"][:],
            sparse_obj["indices"][:],
            sparse_obj["indptr"][:]
        ), shape=shape)
    elif sp_type.startswith("csc"):
        cls_func = sp.csc_matrix if return_matrix else sp.csc_array
        sparr = cls_func((
            sparse_obj["data"][:],
            sparse_obj["indices"][:],
            sparse_obj["indptr"][:]
        ), shape=shape)
    elif sp_type.startswith("coo"):
        cls_func = sp.coo_matrix if return_matrix else sp.coo_array
        sparr = cls_func((
            sparse_obj["data"][:],
            (sparse_obj["row"][:], sparse_obj["col"][:])
        ), shape=shape)
    else:
        raise NotImplementedError(sp_type)

    return sparr


def read_data_grid_from_group(g, return_matrix=True):
    index = g.attrs["batch_names"]
    columns = g.attrs["omics_names"]
    index_str = g.attrs["batch_names_"]
    columns_str = g.attrs["omics_names_"]

    grid = OrderedDict()
    for ri, ri_s in zip(index, index_str):
        g_ri = g[ri_s]
        grid_ri = OrderedDict()
        for ci, ci_s in zip(columns, columns_str):
            if ci_s not in g_ri:
                grid_ri[ci] = None
                continue

            h5_datai = g_ri[ci_s]
            if isinstance(h5_datai, h5py.Group):
                # sparse array
                grid_ri[ci] = read_sparse_in_h5(h5_datai, return_matrix=return_matrix)
            elif isinstance(h5_datai, h5py.Dataset):
                # ndarray
                grid_ri[ci] = h5_datai[:]
        grid[ri] = grid_ri

    return grid


def read_data_grid(fn, key="X", return_matrix=True):
    with h5py.File(fn, "r") as h5:
        g = h5[key]
        grid = read_data_grid_from_group(g, return_matrix=return_matrix)
    return grid


def read_obs_var(fn):
    obs = pd.read_hdf(fn, "obs")
    var = pd.read_hdf(fn, "var")
    return obs, var


def read_nets(fn, return_matrix=True):
    with h5py.File(fn, "r") as h5:
        g = h5["nets"]
        res = {}
        for k, gi in g.items():
            res[k] = read_data_grid_from_group(gi, return_matrix=return_matrix)
    return res
