import numpy as np
from ECO import *
from ECO_original import *
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
def GERCE(M, Niter = 1):
    global databit_num, threshold
    m, n = M.shape
    parity_num = n-databit_num
    H = None
    # cur_rank = 0

    for i in range(Niter):
        Q = np.identity(n, dtype=np.uint8)  # reset ECO mat
        if i == 0: # no permutation for first loop
            R = M
        else:
            R,Q = permute(M,Q) # permute original M

        # perform ECO
        R,Q = ECO(R,Q,BGCE=True) # ECO_original(R,Q,BGCE=True)

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
            # cur_rank = np.linalg.matrix_rank(H) ### rank calculation ###
        else:
            ### rank calculation ###
            # for h in H_formatted:
            #     Htemp = np.append(H, [h], axis = 0)
            #     if np.linalg.matrix_rank(Htemp) > cur_rank: # increase rank
            #         H = np.append(H, [h], axis=0) #add h into H
            #         cur_rank += 1
            ### rank calculation ###

            ### original code ###
            H = np.concatenate((H,H_formatted), axis = 0)
            ### original code ###

        # if H.shape[0] >= parity_num:
        #     return H # stop iteration

    return H
