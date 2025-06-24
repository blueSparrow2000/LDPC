import numpy as np
import numba
from prettyprinter import *

'''
ECO후 얻어야 하는것
ECO matrix E
즉 수행한 column operation들을 모아두어야 한다

'''
@numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices

def ERO(M):
    # ECO를 적용하기 전 row 만 바꾸는 작업 => code word의 순서만 바꾸는 거라 큰 의미 없음

    m,n = M.shape
    i=0
    j=0
    while i < m and j < n:
        if M[i, j] == 0:  # need to fetch 1
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
        i += 1
        j += 1


    return M

def ECO(M):
    # PASS 1
    M = ERO(M)
    M = np.transpose(M) # ECO는 transpose후 ERO를 적용한것과 같다
    m,n = M.shape

    # make a matrix that contains operation
    Q = np.identity(m, dtype=int)  # ECO matrix

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
        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        # update Q => do a column operation too!
        # Q[:, i] = Q[:, i] ^ col
        flip = np.outer(col, Q[i]) # this is more expansive
        Q = Q ^ flip

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








'''
ERO version 
def ERO(M):
    m,n = M.shape

    # make a matrix that contains operation
    Q = np.identity(m, dtype=int)  # ECO matrix

    i=0
    j=0
    while i < m and j < n:
        one_not_found = False
        if M[i, j]==0: # need to fetch 1
            # find value and index of largest element in remainder of column j
            k = np.argmax(M[i:, j]) +i
            if M[k, j]==0: # no 0 found
                one_not_found = True
            else:
                # swap rows
                #M[[k, i]] = M[[i, k]] this doesn't work with numba
                temp = np.copy(M[k])
                M[k] = M[i]
                M[i] = temp

                # update Q (swap Q also)
                temp = np.copy(Q[k])
                Q[k] = Q[i]
                Q[i] = temp

        if not one_not_found: # only if there exists 1 in this diagonal (i,i)
            aijn = M[i, j:]

            col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

            col[i] = 0 #avoid xoring pivot row with itself

            flip = np.outer(col, aijn)

            M[:, j:] = M[:, j:] ^ flip

            # update Q => do a column operation too!
            # Q[:, i] = Q[:, i] ^ col
            flip = np.outer(col, Q[i]) # this is more expansive
            Q = Q ^ flip

        i += 1
        j += 1
        # we may assume i==j

    return M, Q # returns result and ERO matrix

M = np.array([[1,0,0,1],[1,1,1,0],[0,1,0,1]])
R, Q = ERO(M)

print("Result:")
print_arr(R)
print("ECO matrix: ")
print_arr(Q)

'''