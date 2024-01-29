import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef setdiff(int n, int[:] subset):
    cdef Py_ssize_t i, j, k
    cdef int n_subset = subset.shape[0]
    cdef int n_no_subset = n - n_subset

    res = np.zeros(n_no_subset, dtype=np.int64)
    k = 0
    cdef int flag = 0
    for i in range(n):
        flag = 0
        for j in range(n_subset):
            if j == i:
                flag = 1
                break
        if not flag:
            res[k] = i
            k += 1

    return res


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def negative_sample_from_graph(int vnum, int[:] indptr, int[:] indices, int[:] i_neg, vprob):
    """
    该实现是循环地针对每个起点进行采样，采样中只针对其非邻节点采样
    """
    cdef Py_ssize_t n_edge = i_neg.shape[0]
    res = np.zeros(n_edge, dtype=np.intc)

    cdef Py_ssize_t ind
    cdef int i
    for ind in range(n_edge):
        i = i_neg[ind]
        cind_i = indices[indptr[i] : indptr[i + 1]]
        cind_i_neg = setdiff(vnum, cind_i)
        vprob_i_neg = vprob[cind_i_neg]
        j_i_neg = np.random.choice(
            cind_i_neg, p=vprob_i_neg / vprob_i_neg.sum()
        )
        res[ind] = j_i_neg
    return np.array(res)