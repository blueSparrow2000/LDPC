import numpy as np
import numba
'''
완벽한 gaussian elimination이 아니다
column의 1을 모두 제거해서 leading 1 을 남기지 않아야 함
'''
@numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices

# M is a mxn matrix binary matrix
# all elements in M should be uint8
def gf2elim(M,verbose=False):
    m,n = M.shape
    i=0
    j=0

    while i < m and j < n:
        if verbose:
            print("iteration: ", i)
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i
        if M[k,j]==0: #argmax value is 0
            j+=1
            continue
        # swap rows
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp
        if verbose:
            print(M)
            print("ERO")
        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip
        if verbose:
            print(col)
            print(M)

        i += 1
        j += 1

    return M

