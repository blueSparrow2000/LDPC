import numpy as np
import numba
from prettyprinter import *

'''
Sample submatrix of M 
Input) M: matrix , k: number of databits
Out) Sampled submatrix according to sampling rule

'''

def sample_row_col_indices(M,ms,ns):
    m,n = M.shape

    row_survive_idx = np.array([True] * ms + [False] * (m-ms))
    row_indicies = np.arange(m)

    col_survive_idx = np.array([True] * ns + [False] * (n-ns))
    col_indicies = np.arange(n)

    sample_row_idx = row_indicies[np.random.permutation(row_survive_idx)]
    sample_col_idx = col_indicies[np.random.permutation(col_survive_idx)]
    # print(ms, ns)
    # print(sample_row_idx,sample_col_idx)
    return sample_row_idx, sample_col_idx


def sample_col_indices(M,ns):
    m,n = M.shape

    col_survive_idx = np.array([True] * ns + [False] * (n-ns))
    col_indicies = np.arange(n)

    sample_col_idx = col_indicies[np.random.permutation(col_survive_idx)]
    return sample_col_idx


def sample_submatrix(M,sample_row_idx,sample_col_idx):
    submatrix = M[np.ix_(sample_row_idx,sample_col_idx)]

    return submatrix





