import numpy as np
import scipy.sparse as scsparse
from gauss_elim import *
np.random.seed(seed=1)


'''
fast rank computation
https://stackoverflow.com/questions/56856378/fast-computation-of-matrix-rank-over-gf2


fast matmul on GF2 (binary matrix)
https://github.com/malb/m4ri?tab=readme-ov-file
'''


'''
Returns H matrix and a code word
pooling_factor: ratio of 'code word length' with 'code word amount'
noise_level: amount of noise that simulates bit shift

'''


@numba.njit(fastmath=True, cache=True, parallel=True)
def matmul_f2(m1, m2):
    mr = np.empty((m1.shape[0], m2.shape[1]), dtype=np.bool_)
    for i in numba.prange(mr.shape[0]):
        for j in range(mr.shape[1]):
            acc = False
            for k in range(m2.shape[0]):
                acc ^= m1[i, k] & m2[k, j]
            mr[i, j] = acc
    return mr.astype(np.int64)

def sample_LDPC(n,k, pooling_factor = 2,noise_level = 0):
    # build a generator matrix
    I = np.identity(k, dtype=int)
    P = scsparse.random(k, n-k, density=0.25, data_rvs=np.ones)
    P = P.toarray().astype(np.int64)
    generator = np.concatenate((I, P), axis=1) # generator mat

    # sample PCM
    I_prime = np.identity(n-k, dtype=int)
    H = np.concatenate((P.T, I_prime), axis=1) # PCM

    #sample message bits - each row is a message bit
    M = pooling_factor*n # 10 messages
    message_bits = np.random.choice([0, 1], size=(M,k), p=[1./2, 1./2]) #  np.identity(k, dtype=int)

    ################################## matmul needs to be faster only here ###############################################
    code_words = matmul_f2(message_bits, generator) # message_bits@generator
    ################################## matmul needs to be faster only here ###############################################

    message_rank = np.linalg.matrix_rank(message_bits)
    code_rank = np.linalg.matrix_rank(code_words)
    if k > code_rank:
        if k> message_rank:
            print(message_rank)
            print("[WARNING] message rank too small! need more data")
        else:
            print(code_rank)
            print("[WARNING] generator matrix produces degenerate code!")

    if noise_level: # add gaussian noise to code_words matrix
        # apply noise
        return H, code_words

    return H, code_words


# H, A = sample_LDPC(5,3)
# A = A.T
# print(A)
#
# R = gf2elim(A)
# print(R)


