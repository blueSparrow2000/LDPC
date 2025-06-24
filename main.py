from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from sparsifyer import *

def check_success(target_array,my_array):
    return (target_array == my_array).all()

# sample LDPC code word
H, A = sample_LDPC(500,350)
print("H matrix: ")
print_arr(H)
print("Code word matrix: ")
print_arr(A)

# 1. apply ECO to A(code word matrix) and get Q, R
R,Q = ECO(A)
print("Result:")
print_arr(R)
print("ECO matrix: ")
print_arr(Q)

# 2. search for sparse columns in R and get columns in Q of the corresponding indices
idx = get_sparse_column_idx(R)
H_recovered = get_sparse_columns(Q,idx).T
print("recovered H matrix row space")
print_arr(H_recovered)

print("Success?: ", check_success(H,H_recovered))
# 3. after collecting n-k such columns of Q, transpose it to form H

# 4. Sparsify H matrix -> see if it has diagonal or bi-diagonal format















