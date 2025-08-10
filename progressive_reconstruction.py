from prettyprinter import *
from gauss_elim import gf2elim
from LDPC_sampler import *
from extracter import *
from verifier import *
from variables import *
from submatrix_sampling import *
from dubiner_sparsifyer import sparsify


import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''

##### sample LDPC code word

H, A = sample_LDPC(codeword_len,databit_num,pooling_factor=pooling_factor,noise_level=noise_level)

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)

    # print("Code word matrix: ")
    # print_arr(A)
else:
    print("Code word generated")

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))


##### submatrix sampling : progressive reconstruction of ldpc code 논문의 fig2를 보면 이렇게 잡는다
col_factor=0.5 # 0.5
row_factor=0.5 # 0.3
ns = round(col_factor * codeword_len)  # number of sampled col
ms = round(row_factor * ns)  # number of sampled row


row_indices, col_indices = sample_row_col_indices(A,ms,ns)
# print(col_indices)
sub = sample_submatrix(A,row_indices, col_indices)
#print_arr(sub)

# gaussian elimination 으로 복원하는 방법을 사용함
Gs = gf2elim(sub)
P = Gs[:,ms:]
P_t = np.transpose(P)
Hs = np.concatenate((P_t, np.identity(ns-ms, dtype=int)), axis=1)
# print("Hs")
# print_arr(Hs)

# dubiner sparsification
Hr = sparsify(Hs, ms,ns)
mr,nr = Hr.shape

# Hs를 다시 H의 규격에 맞게(길이 n) 0을 추가해서 늘려줌
H_recovered = np.zeros((mr,codeword_len), dtype=int) # H_recovered = np.zeros((ns-ms,codeword_len), dtype=int)
H_recovered[:,col_indices] = Hr

this_time = time.time()
if not LARGE_CODE:
    print("Recovered H matrix: ")
    print_arr(H_recovered)
else:
    print("SVR done")
print("Elapsed time: %s seconds" % round(this_time - start_time,3))

print("Success?: ", check_success(H,H_recovered))



