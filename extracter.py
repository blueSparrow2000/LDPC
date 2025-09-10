import numpy as np
from matrix_mul import matmul_f2
''' 
threshold: number of 1's that is acceptable - rule of thumb to 0.65*(M-n)/2 considering noise rate
'''
def get_sparse_column_idx(A, threshold = 0):
    # for each column, count number of ones
    counts = A.sum(0)
    sparse_check = np.array([True if x <= threshold else False for x in counts])
    # print(sparse_check)
    return sparse_check

def get_sparse_columns(Q,idx):
    idx = idx.reshape(Q.shape[0])
    return Q[:,idx==True]

'''
For each dual vectors, detect errors with original code word matrix
Then sort the matrix with respect to number of errors, and collect top n-k dual vectors with the least error count
'''
def reliability_extraction(H,codeword, dual_vector_num):
    d_num, n= H.shape
    if d_num <= dual_vector_num: # found less dual vectors
        # perform only if dual vectors are more than n-k
        return H

    error_detected = matmul_f2(H,codeword.T)
    num_err_each_row = np.count_nonzero(error_detected, axis=1)
    H_sorted = H[num_err_each_row.argsort(),:]

    # print("H matrix to extract row")
    # print(H)
    #print("H * Codeword")
    #print(error_detected)
    #print("Error number for each row")
    #print(num_err_each_row)

    return H_sorted[:dual_vector_num, :]


'''
Calculate reliability (num of errors) 
for each block, assign one reliability constant 
hence, check reliability w.r.t a block
'''
def get_confidence_reliability(H,codeword, dual_vector_num, Z): # Z: block size
    d_num, n= H.shape
    if d_num <= dual_vector_num: # found less dual vectors
        # perform only if dual vectors are more than n-k
        return H

    error_detected = matmul_f2(H,codeword.T)
    num_err_each_row = np.count_nonzero(error_detected, axis=1)
    H_sorted = H[num_err_each_row.argsort(),:]

    return H_sorted[:dual_vector_num, :]









