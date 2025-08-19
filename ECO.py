import numpy as np
import numba
from prettyprinter import *

'''
ECO후 얻어야 하는것
ECO matrix E
즉 수행한 column operation들을 모아두어야 한다
'''

@numba.jit(nopython=True, parallel=True)
def ECO(M,Q, BGCE = True): #Q = None,
    # 1: row swap
    m1, n1 = M.shape
    i = 0
    j = 0
    while i < m1 and j < n1:
        if M[i, j] == 0:  # need to fetch 1
            # find value and index of largest element in remainder of column j
            k = np.argmax(M[i:, j]) + i
            if M[k, j] == 0:  # argmax value is 0
                i += 1
                j += 1
                continue
            # swap rows
            # M[[k, i]] = M[[i, k]] this doesn't work with numba
            temp = np.copy(M[k])
            M[k] = M[i]
            M[i] = temp
        i += 1
        j += 1

    M = np.transpose(M) # ECO는 transpose후 ERO를 적용한것과 같다
    m,n = M.shape

    # make a matrix that contains operation
    # if Q is None:
    #     Q = np.identity(m, dtype=int)  # ECO matrix
    # Q = np.identity(m, dtype=int)  # ECO matrix

    # 2: column swap and elimination
    i=0
    j=0
    while i < m and j < n:
        if M[i, j]==0: # need to fetch 1
            # find value and index of largest element in remainder of column j
            k = np.argmax(M[i:, j]) +i
            if M[k, j] == 0:  # argmax value is 0
                j += 1
                continue

            # swap rows
            #M[[k, i]] = M[[i, k]] this doesn't work with numba
            temp = np.copy(M[k])
            M[k] = M[i]
            M[i] = temp

            # update Q (swap Q also)
            temp = np.copy(Q[k])
            Q[k] = Q[i]
            Q[i] = temp

        # only if there exists 1 in this diagonal (i,i)
        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself
        if not BGCE:# only GCE, not bidirectional
            col[:i] = 0

        ########################## numpy optimizer(slow) #############################
        # aijn = M[i, j:]
        # flip = np.outer(col, aijn)
        # M[:, j:] = M[:, j:] ^ flip
        #
        # # update Q => do a column operation too!
        # qijn = Q[i,:]
        # flip2 = np.outer(col, qijn) # this is more expansive
        # Q = Q ^ flip2
        ########################## numpy optimizer #############################

        ######################### MANUAL FOR LOOP #########################
        # safe parallelism
        for ix in numba.prange(m):
            if col[ix]:
                M[ix, j:] ^= M[i, j:] # flip bit
                Q[ix, :] ^= Q[i,:]

        # unstable?
        # for ix in numba.prange(m):
        #     if col[ix]:
        #         for jx in numba.prange(j,n):
        #             M[ix, jx] ^= M[i, jx] # flip bit
        #         for jx in numba.prange(m):
        #             Q[ix, jx] ^= Q[i,jx]

        ######################### MANUAL FOR LOOP #########################
        i += 1
        j += 1
        # we may assume i==j


    # transpose to get column operation matrix and resulting matrix
    M = np.transpose(M)
    Q = np.transpose(Q)
    return M, Q # returns result and ECO matrix
#
# M = np.array([[1,0,0,1],[1,1,1,0],[0,1,0,1]])
# R, Q = ECO(M)
#
# print("Result:")
# print_arr(R)
# print("ECO matrix: ")
# print_arr(Q)







