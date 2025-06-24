import numpy as np


def print_arr(M):
    m, n = M.shape
    for i in range(m):
        print(' '.join(map(str, M[i])))
