import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA

from .. import typehint as typ


def tfidf(X: typ.DATA_ELEM) -> typ.DATA_ELEM:
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
        tf = X.multiply(1 / X.sum(axis=1)[:, None])
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
