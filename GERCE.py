import numpy as np
from ECO import *
from extracter import *
from variables import threshold, parity_num,databit_num
from formatter import diag_format

def permute(M,Q):
    m, n = M.shape

    permute_idx = np.random.permutation(n)
    M = M[:,permute_idx]
    Q = Q[:,permute_idx]
    return M,Q


'''
Not tested. Should test this function!
'''
def GERCE(M, Niter = 1,threshold=0,databit_num=0):

    m, n = M.shape
    parity_num = n-databit_num
    H = None

    for i in range(Niter):
        Q = np.identity(n, dtype=np.int64)  # reset ECO mat
        if i == 0: # no permutation for first loop
            R = M
        else:
            R,Q = permute(M,Q) # permute original M

        # perform ECO
        R,Q = ECO(R,Q,BGCE=True)

        # extract sparse col
        idx = get_sparse_column_idx(R, threshold)

        # 3. after collecting n-k such columns of Q, transpose it to form H
        H_recovered = get_sparse_columns(Q, idx).T

        # 4. extract n-k dual vectors if more than one vector is considered to be sparse
        # H_extracted = reliability_extraction(H_recovered, M, parity_num) # reliability extraction 에는 원래의 행렬을 넣어야 함!
        H_extracted = H_recovered # skip reliability extraction (since it costs a lot)

        ############################### Formatting is different for QC LDPC codes #############################
        H_formatted = diag_format(H_extracted, databit_num)
        ############################### Formatting is different for QC LDPC codes #############################

        if H is None: # first loop
            H = H_formatted
        else:
            H = np.concatenate((H,H_formatted), axis = 0)
        # if H.shape[0] >= parity_num:
        #     return H # stop iteration

    return H
