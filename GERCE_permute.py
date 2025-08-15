from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from formatter import *
from variables import *
from GERCE import GERCE

import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''

# sample LDPC code word

H, A = sample_LDPC(codeword_len,databit_num,density = density,pooling_factor=pooling_factor,noise_level=noise_level)

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)

    print("Code word matrix: ")
    print_arr(A)
else:
    print("Code word generated")

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))


# 1. apply ECO to A(code word matrix) and get Q, R
H_formatted = GERCE(A, Niter = 1)

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))

H_padded = H_formatted
# padding H
# mh, nh = H_formatted.shape
# H_padded = np.zeros((mh, n), dtype=np.int64)  # H_recovered = np.zeros((ns-ms,codeword_len), dtype=int)
# H_padded[:, col_idx] = H_formatted

this_time = time.time()
print("Success?: ", check_success(H,H_padded))
print("Elapsed time: %s seconds" % round(this_time - start_time,3))






















