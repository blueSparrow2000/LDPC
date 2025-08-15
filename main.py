from prettyprinter import *
from ECO import *
from ECO_original import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from formatter import *
from variables import *

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
Q_aux = np.identity(codeword_len, dtype=np.int64)
R,Q = ECO(A,Q_aux,BGCE=BGCE)#ECO_original(A,Q_aux,BGCE=BGCE)
if not LARGE_CODE:
    print("Result:")
    print_arr(R)
    print("ECO matrix: ")
    print_arr(Q)
else:
    print("ECO complete")

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))

# 2. search for sparse columns in R and get columns in Q of the corresponding indices
idx = get_sparse_column_idx(R,threshold)

# 3. after collecting n-k such columns of Q, transpose it to form H
H_recovered = get_sparse_columns(Q,idx).T

# 4. extract n-k dual vectors if more than one vector is considered to be sparse
H_extracted = reliability_extraction(H_recovered, A,parity_num)

# 5. Sparsify/Format H matrix -> see if it has diagonal or bi-diagonal format
H_formatted = diag_format(H_extracted, databit_num)

if not LARGE_CODE:
    print("recovered H matrix row space")
    print_arr(H_extracted)
    print("formatted H")
    print_arr(H_formatted)

print("Success?: ", check_success(H,H_formatted))
print("Elapsed time: %s seconds" % round(time.time() - start_time,3))





















