from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from formatter import *
from variables import *
from GERCE import GERCE
from submatrix_sampling import sample_col_indices

import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)

- permuting does not help recovering
'''

# sample LDPC code word

H, A = sample_LDPC(codeword_len,databit_num,density = density,pooling_factor=pooling_factor,noise_level=noise_level)

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)
else:
    print("Code word generated")

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))


################### SAMPLE COL IDX ####################
v = round(codeword_len) - parity_num - 4 #parity_num - 6 #
kappa = 0
H_final = None

for i in range(10):
    col_indices = sample_col_indices(A, v, always_sample_parity_bits=parity_num) #  : always sample last n-k bits
    # print(col_indices)
    Mv = np.copy(A[:, col_indices])


    # 1. apply ECO to A(code word matrix) and get Q, R
    H_formatted = GERCE(Mv, Niter = 1)
    mv, nv = H_formatted.shape

    # zero pad H to form PCM in dim n space
    H_pad = np.zeros((mv, codeword_len), dtype=np.uint8)  # H_recovered = np.zeros((ns-ms,codeword_len), dtype=int)
    H_pad[:, col_indices] = H_formatted

    for j in range(mv):  # for each recovered dual vectors
        h = H_pad[j]  # a dual vector
        # print_arr(h)
        if H_final is None:
            H_final = np.array([h])  # initial vector
            kappa += 1
            continue
        Htemp = np.append(H_final, [h], axis=0)
        if np.linalg.matrix_rank(Htemp) > kappa:  # increase rank
            H_final = np.append(H_final, [h], axis=0)  # add h into H
            kappa += 1

# do a reliablity extraction again
H_final = reliability_extraction(H_final, A,parity_num)
H_final = diag_format(H_final, databit_num)  # do another formating
this_time = time.time()
print("Success?: ", check_success(H,H_final))
print_arr(H_final)
# print("Elapsed time: %s seconds" % round(this_time - start_time,3))






















