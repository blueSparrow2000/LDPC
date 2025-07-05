'''
This is a full code which should be translated into C for performance
'''

import scipy.sparse as scsparse
import numpy as np
import numba
np.random.seed(seed=1)


codeword_len = 10
LARGE_CODE = True if codeword_len > 50 else False
databit_num = 9
parity_num = codeword_len-databit_num
noise_level=2
pooling_factor = 2
threshold = ((pooling_factor-1)*codeword_len)*0.325  # suggested beta coeff on the paper

def print_arr(M):
    m, n = M.shape
    for i in range(m):
        print(' '.join(map(str, M[i])))

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

    if noise_level==1:  # add one bit noise
        code_words[0,-1] = not code_words[0,-1] # add noise to only the last bit of the first row
        #code_words[-1, - 1] = not code_words[-1,- 1]  # add noise to only the last bit of the last row
        return H, code_words
    elif noise_level==2: # add two bit noise
        code_words[0,-1] = not code_words[0,-1] # add noise to only the last bit of the first row
        code_words[-1, - 1] = not code_words[-1,- 1]  # add noise to only the last bit of the last row
        return H, code_words
    elif noise_level==10: # add gaussian noise to code_words matrix
        return H, code_words
    return H, code_words


@numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices
def gf2elim(M,verbose=False):
    m,n = M.shape
    i=0
    j=0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i
        if M[k,j]==0: #argmax value is 0
            j+=1
            continue
        # swap rows
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp
        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip
        i += 1
        j += 1

    return M

@numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices
#@numba.njit(fastmath=True, cache=True,nopython=True, parallel=True)

def row_swap(M):
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
            temp = np.copy(M[k])
            M[k] = M[i]
            M[i] = temp
        i += 1
        j += 1
    return M

def ECO(M):
    # PASS 1
    M = row_swap(M)
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
        flip = np.outer(col, Q[i]) # this is more expansive
        Q = Q ^ flip

        i += 1
        j += 1

    # transpose to get column operation matrix and resulting matrix
    M = np.transpose(M)
    Q = np.transpose(Q)
    return M, Q # returns result and ECO matrix

def get_sparse_column_idx(A, threshold):
    # for each column, count number of ones
    counts = A.sum(0)
    sparse_check = np.array([True if x <= threshold else False for x in counts])
    return sparse_check

def get_sparse_columns(Q,idx):
    idx = idx.reshape(Q.shape[0])
    return Q[:,idx==True]

def reliability_extraction(H,codeword, dual_vector_num):
    d_num, n= H.shape
    if d_num <= dual_vector_num: # found less dual vectors
        # perform only if dual vectors are more than n-k
        return H
    error_detected = matmul_f2(H,codeword.T)
    num_err_each_row = np.count_nonzero(error_detected, axis=1)
    H_sorted = H[num_err_each_row.argsort(),:]
    return H_sorted[:dual_vector_num, :]


def check_success(target_array,my_array):
    tx,ty = target_array.shape
    mx,my = my_array.shape
    if tx > mx: # too small!
        print("Dual vectors missing: ", tx-mx)
        return False
    return (target_array == my_array).all()

'''
Format given matrix to H matrix form 
which has identity matrix at the last n-k columns
'''
def diag_format(H, databit_num):
    p,n = H.shape
    H_before = np.concatenate((H[:, databit_num:],H[:,:databit_num]), axis=1)
    H_gauss = gf2elim(H_before)
    result = np.concatenate(( H_gauss[:, n-databit_num:],H_gauss[:, :n-databit_num]), axis=1)  #
    return result


H, A = sample_LDPC(codeword_len,databit_num,pooling_factor=pooling_factor,noise_level=noise_level)

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)

    print("Code word matrix: ")
    print_arr(A)
else:
    print("Code word generated")

# 1. apply ECO to A(code word matrix) and get Q, R
R,Q = ECO(A)
if not LARGE_CODE:
    print("Result:")
    print_arr(R)
    print("ECO matrix: ")
    print_arr(Q)
else:
    print("ECO complete")

# 2. search for sparse columns in R and get columns in Q of the corresponding indices
idx = get_sparse_column_idx(R,threshold)
H_recovered = get_sparse_columns(Q,idx).T
H_extracted = reliability_extraction(H_recovered, A,parity_num)
H_formatted = diag_format(H_extracted, databit_num)

if not LARGE_CODE:
    print("recovered H matrix row space")
    print_arr(H_extracted)
    print("formatted H")
    print_arr(H_formatted)

print("Success?: ", check_success(H,H_formatted))






