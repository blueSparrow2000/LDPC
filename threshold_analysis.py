from ECO import *
from ECO_original import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from formatter import *
from variables import *
from csv_writter import save_recovery_data_row_csv,init_header
import time
start_time = time.time()

'''
Objective 1.

Fixing pooling factor to 7,
find the best threshold value = beta (threshold = codewords/2*beta)

using meshgrid search

'''
global threshold, pooling_factor, codeword_len

SEEDTESTAMOUNT = 1 # 1로 하면 하나의 case에 대해서만 수행한다
SEEDOFFSET = 6
beta_list = np.arange(0.25,0.75,0.1) #[0 + 0.01*i for i in range(100)]
init_header(["threshold","found ratio","recover ratio" ])


for i in range(len(beta_list)):
    beta = beta_list[i]
    threshold = round(((pooling_factor-1)*codeword_len)*beta/2)

    found_ratio = 0
    recovery_ratio = 0
    for seed_num in range(SEEDOFFSET,SEEDTESTAMOUNT+SEEDOFFSET):
        ############################### code effected by SEED ##############################
        np.random.seed(seed=seed_num)
        # sample LDPC code word
        H, A = sample_LDPC(codeword_len, databit_num, density=density, pooling_factor=pooling_factor,
                           noise_level=noise_level)
        tx, ty = H.shape

        # 1. apply ECO to A(code word matrix) and get Q, R
        Q_aux = np.identity(codeword_len, dtype=np.uint8)
        R = np.copy(A)
        R, Q = ECO(R, Q_aux, BGCE=BGCE)  # ECO_original(A,Q_aux,BGCE=BGCE)
        ############################### code effected by SEED ##############################

        # print("M%d ECO complete"%i, end=' - ')
        # print(" %s seconds" % round(time.time() - start_time,3))

        # 2. search for sparse columns in R and get columns in Q of the corresponding indices
        idx = get_sparse_column_idx(R,threshold)

        # 3. after collecting n-k such columns of Q, transpose it to form H
        H_recovered = get_sparse_columns(Q,idx).T

        # 4. extract n-k dual vectors if more than one vector is considered to be sparse
        H_extracted = reliability_extraction(H_recovered, A,parity_num)

        # 5. Sparsify/Format H matrix -> see if it has diagonal or bi-diagonal format
        H_formatted = diag_format(H_extracted, databit_num)

        # get the correct guess data
        mx,my = H_formatted.shape

        correct_guess = 0
        for i in range(tx):
            if (H_formatted == H[i]).all(axis=1).any():
                correct_guess += 1

        found_ratio += round(100*mx/tx, 1)/SEEDTESTAMOUNT
        recovery_ratio += round(100*correct_guess/tx, 1)/SEEDTESTAMOUNT

    datarow = [beta,round(found_ratio,1) ,round(recovery_ratio,1) ]

    save_recovery_data_row_csv(datarow)

    # print("Success?: ", check_success(H,H_formatted))
    # print("Total elapsed time: %s seconds" % round(time.time() - start_time,3))





