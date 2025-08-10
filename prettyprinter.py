import numpy as np


def print_arr(M):
    if len(list(M.shape))==1: # 1D array
        print(' '.join(map(str, M)))
        return

    m, n = M.shape
    for i in range(m):
        print(' '.join(map(str, M[i])))
