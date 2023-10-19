from typing import OrderedDict as odict
from typing import (
    Optional, Tuple, Union, TypedDict, Dict, Sequence, Mapping, List, Any
)
# from typing_extensions import OrderedDict

from pandas import DataFrame
from numpy import ndarray
# from numpy.random import Generator
from scipy import sparse as sp
import torch
from torch.distributions import Distribution


# 1. For DataGrid Class
SPARSE = Union[sp.coo_array, sp.csr_array, sp.csc_array]
DATA_ELEM = Union[ndarray, SPARSE, Tuple[int, int]]
DATA_MAPPING = Union[Mapping[str, DATA_ELEM],
                     Mapping[str, Mapping[str, DATA_ELEM]]]
DATA_LIST = Sequence[Optional[DATA_ELEM]]
DATA_RECORD = Sequence[Tuple[str, str, DATA_ELEM]]
DATA_GRID_INIT = Optional[Union[DATA_MAPPING, DATA_LIST, DATA_RECORD]]

GRID_INDEX_ONE = Union[str, int, slice]
GRID_INDEX = Union[
    GRID_INDEX_ONE,
    Tuple[GRID_INDEX_ONE, GRID_INDEX_ONE],
    Tuple[GRID_INDEX_ONE, GRID_INDEX_ONE, GRID_INDEX_ONE],
    Tuple[GRID_INDEX_ONE, GRID_INDEX_ONE, GRID_INDEX_ONE, GRID_INDEX_ONE]
]

META_ODICT = odict[str, DataFrame]
NUM_FEATS = odict[str, int]
NETWORKS = Dict[str, sp.coo_array]

T = torch.Tensor
TUPLE_NET = Tuple[
    Tuple[int, int],  # shape
    ndarray,          # i
    ndarray,          # j
    ndarray,          # sign
    ndarray,          # wt
]
NETS = Dict[str, Tuple[T, T, T]]

# NOTE: numpy有两套定义随机性的系统：randomstate和generator
# 因为sklearn用的是randomstate，所以我们也是沿用randomstate
SEED = Optional[int]

# 2. For Model
EMBED = Union[T, tuple[T, ...]]  # 有可能返回多个隐变量
X_REC = odict[str, Distribution]
LOSS = Dict[str, T]  # 这里储存的是loss和其对应的weight
LOSS_W = Dict[str, Tuple[T, float]]
FORWARD = Dict[str, Union[Distribution, T]]

# For Encoder
# T = LOSS = torch.Tensor
# LOSSES = odict[str, LOSS]
# Z = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
# REC = odict[str, torch.Tensor]
#
# DIST = Distribution
# DISTS = odict[str, DIST]
# RECDIST = odict[str, Tuple[Distribution, TENSOR]]
#
# DIFFI = OrderedDict[str, TENSOR]
# DIFF_RES = Tuple[DIFFI, DIFFI, DIFFI]
#
# WEIGHT = Union[float, Sequence[float]]


class SLICE(TypedDict, total=False):
    input: odict[str, T]
    output: odict[str, T]  # outputs(counts)
    mask: odict[str, T]
    blabel: T
    dlabel: T
    sslabel: T
    imp_blabel: odict[str, T]
    sample_graph: Tuple[T, T, T]
    # label: Union[str, int]


class BATCH(TypedDict, total=False):
    ind: List[Any]
    input: odict[str, T]
    output: odict[str, T]  # outputs(counts)
    mask: odict[str, T]
    blabel: T
    dlabel: T
    sslabel: T
    label: Union[str, int]
    imp_blabel: odict[str, T]
    varp: NETS
    sample_graph: Tuple[T, T, T]
    walk: Tuple[T, T]
    pos_sign: T
