from prettyprinter import *
from LDPC_sampler import *
from verifier import *
from variables import *
from submatrix_sampling import sample_col_indices
from dubiner_sparsifyer import sparsify
from GERCE import GERCE

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


kappa = 0 # num_dual_vectors
Niter = 0
v = round(codeword_len/2) # codeword_len #  # col_extract_num
vmin = 0
H_final = None  # currently found dual vectors

Nmin = 100 # parameter determined by v, n and BER (more than 100)

total_iteration = 1 # full iteration number

while Niter<total_iteration:
    kappa_old = kappa

    for i in range(Nmin):
        col_indices = sample_col_indices(A, v)
        Mv = A[:, col_indices]
        Hv = GERCE(Mv,threshold = threshold, databit_num = databit_num)
        mv,nv = Hv.shape

        # print("sampled columns")
        # b = np.zeros(shape=(codeword_len,), dtype=np.int8)
        # b[col_indices] = 1
        # print_arr(b)

        # zero pad Hv to form PCM in dim n space
        H_pad = np.zeros((mv, codeword_len), dtype=np.int8)  # H_recovered = np.zeros((ns-ms,codeword_len), dtype=int)
        H_pad[:, col_indices] = Hv


        for j in range(mv): # for each recovered dual vectors
            h = H_pad[j] # a dual vector
            # print_arr(h)
            if H_final is None:
                H_final = np.array([h]) # initial vector
                kappa += 1
                continue
            Htemp = np.append(H_final, [h], axis = 0)
            if np.linalg.matrix_rank(Htemp) > kappa: # increase rank
                H_final = np.append(H_final, [h], axis=0) #add h into H
                kappa += 1

    if Niter == 0: # first loop
        if kappa_old < kappa: # if new vector obtained
            Niter += 1
            vmin = v
        else: # if no new vector is obtained
            v = round(v/2)
            print("First pass adjusting v: ", v)
            if v<= 2:
                print("Failed to recover")
                break #
            continue
    else:
        if kappa_old < kappa:
            v = round((v+codeword_len/2)/2)
        else:
            v = round((v + vmin / 2) / 2)
            print("Not found, adjusting v: ", v)
            if v<=vmin:
                print("Failed to recover")
                break #

    # dubiner sparsification
    # mf,nf = H_final.shape
    # H_final = sparsify(H_final, mf,nf)

    # decoding if possible
    Niter += 1

this_time = time.time()
if not LARGE_CODE:
    print("Recovered H matrix: ")
    print_arr(H_final)
else:
    print("SVR done")
print("Elapsed time: %s seconds" % round(this_time - start_time,3))
print("Success?: ", check_success(H,H_final))
