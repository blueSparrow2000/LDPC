from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from variables import *
from formatter import *
from GERCE import GERCE

import time
start_time = time.time()
################################## sample LDPC code ##########################################

n = 10
k = 7


I = np.identity(7, dtype=np.int64)

########## custom generator component P: k x (n-k)
P = np.array([[1,0,0], [0,1,0], [1,0,0],[0,1,0],[0,0,1],[0,0,1],[0,0,0]])

generator = np.concatenate((I, P), axis=1) # generator mat

# sample PCM
I_prime = np.identity(n-k, dtype=np.int64)
H = np.concatenate((P.T, I_prime), axis=1) # PCM

#sample message bits - each row is a message bit
M = 2*n # 10 messages
message_bits = np.random.choice([0, 1], size=(M,k), p=[1./2, 1./2]) #  np.identity(k, dtype=int)

code_words = matmul_f2(message_bits, generator) # message_bits@generator

print("H matrix given: ")
print_arr(H)
################################## sample LDPC code ##########################################

### sampling column index
col_idx = [0,1,2,3,4,5,6,7,8,9]#[0,1,2,3,8] #

sampled_code = code_words[:, col_idx]
# print("code word given: ")
# print_arr(code_words)
# print("Sampled code word: ")
# print_arr(sampled_code)

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))


######################### Process LDPC code ###########################
H_formatted = GERCE(sampled_code, 1) # databut_num = 7 / threshold = 3
# Q_aux = np.identity(codeword_len, dtype=np.int64)
# R,Q = ECO(sampled_code,Q_aux)
#
# # print("Result:")
# # print_arr(R)
# # print("ECO matrix: ")
# # print_arr(Q)
# print("Elapsed time: %s seconds" % round(time.time() - start_time,3))
#
# # 2. search for sparse columns in R and get columns in Q of the corresponding indices
# idx = get_sparse_column_idx(R,n*0.325)
#
# # 3. after collecting n-k such columns of Q, transpose it to form H
# H_recovered = get_sparse_columns(Q,idx).T
#
# # 4. extract n-k dual vectors if more than one vector is considered to be sparse
# H_extracted = reliability_extraction(H_recovered, sampled_code,3)
#
# # 5. Sparsify/Format H matrix -> see if it has diagonal or bi-diagonal format
# H_formatted = diag_format(H_extracted, k)

######################### Process LDPC code ###########################

# padding H
mh, nh = H_formatted.shape
H_padded = np.zeros((mh, n), dtype=np.int64)  # H_recovered = np.zeros((ns-ms,codeword_len), dtype=int)
H_padded[:, col_idx] = H_formatted
# H_padded = H_formatted

this_time = time.time()
print("Recovered H matrix: ")
print_arr(H_padded)
print("Elapsed time: %s seconds" % round(this_time - start_time,3))
print("Success?: ", check_success(H,H_padded))







