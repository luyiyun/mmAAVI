from typing import Tuple
import warnings

import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from mudata import MuData

from . import typehint as typ
from .utils import merge_multi_obs_cols


def log1p_norm(dat: Tuple[np.ndarray, sp.csr_matrix]) -> np.ndarray:
    EPS = 1e-7

    if isinstance(dat, np.ndarray):
        dat = np.log1p(dat)
    elif sp.issparse(dat):
        dat = dat.log1p()
        dat = dat.todense()
    dat = (dat - dat.mean(axis=0)) / (dat.std(axis=0) + EPS)
    return dat


def tfidf(
    X: Tuple[np.ndarray, sp.csr_matrix]
) -> Tuple[np.ndarray, sp.csr_matrix]:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(dat: typ.DATA_ELEM, n_components: int = 20, **kwargs) -> np.ndarray:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        # Keep deterministic as the default behavior
        kwargs["random_state"] = 0
    X = tfidf(dat)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    return X_lsi


def pca(dat: typ.DATA_ELEM, n_components: int = 20, **kwargs) -> np.ndarray:
    res = PCA(n_components, **kwargs).fit_transform(dat)
    return res


def merge_obs_from_all_modalities(mdata: MuData, key: str) -> str:
    if key in mdata.obs.columns:
        # if finding batch label，directly use
        return

    assert all(key in adatai.obs.columns for adatai in mdata.mod.values()), (
        f"{key} is neither in mudata.obs nor "
        "in adata.obs for each modality."
    )
    # if not finding, create one by omics
    if key in mdata.obs.columns:
        warnings.warn(
            f"{key} exists, it will " "be cover by intermediate columns."
        )
    array = merge_multi_obs_cols(
        [np.asarray(mdata.obs[f"{k}:{key}"].values) for k in mdata.mod.keys()]
    )
    mdata.obs[key] = array
