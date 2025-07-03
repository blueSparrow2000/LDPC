from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from sparsifyer import *
from verifier import *
'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''

# sample LDPC code word
codeword_len = 10
databit_num = 7
parity_num = codeword_len-databit_num
noise_level=1
pooling_factor = 2
threshold = ((pooling_factor-1)*codeword_len)*0.325  # suggested beta coeff on the paper
# print(threshold)
H, A = sample_LDPC(codeword_len,databit_num,pooling_factor=pooling_factor,noise_level=noise_level)
print("H matrix: ")
print_arr(H)
# print("Code word matrix: ")
# print_arr(A)

# 1. apply ECO to A(code word matrix) and get Q, R
R,Q = ECO(A)
print("Result:")
print_arr(R)
print("ECO matrix: ")
print_arr(Q)

# 2. search for sparse columns in R and get columns in Q of the corresponding indices
idx = get_sparse_column_idx(R,threshold)
H_recovered = get_sparse_columns(Q,idx).T
H_extracted = reliability_extraction(H_recovered, A,parity_num)
print("recovered H matrix row space")
print_arr(H_extracted)

H_formatted = diag_format(H_extracted, databit_num)
print("formatted H")

print_arr(H_formatted)
print("Success?: ", check_success(H,H_formatted))
# 3. after collecting n-k such columns of Q, transpose it to form H

# 4. Sparsify H matrix -> see if it has diagonal or bi-diagonal format



















