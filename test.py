import numpy as np
from data_saver import *
from csv_writter import *
'''
# difference comparison
A = read_matrix('decoded_codeword')
B = read_matrix('error_free_codeword')
C = read_matrix('noisy_codeword')
# diff_origin = B^C
# save_image_data(diff_origin, filename="noise_map")
diff = A^B

def remove_error(A,diff):
    delete_indices = []
    error_by_row = np.sum(diff, axis=1)
    l = len(error_by_row)
    for i in range(l):
        if error_by_row[i]>0: # more than x errors
            delete_indices.append(i) # get the index of the row

    for i in range(len(delete_indices)):
        el = delete_indices.pop()
        A = np.delete(A, (el), axis=0) # delete row

    # saved error-free version
    save_matrix(A, filename='decoded_codeword')



save_error_image(A, diff, mode = 'blob')

# remove_error(A,diff)
'''
from prettyprinter import *
from LDPC_sampler import *
from QCLDPC_sampler import *
from submatrix_sampling import *
from dubiner_sparsifyer import sparsify
from block_recover import *
from LDPC_sampler import *
from verifier import *
from formatter import *
from data_saver import *
from csv_writter import *
from bit_flip_decoder_sequential import ldpc_bitflip_seqdecode_numba

from numba import njit, prange

import time
start_time = time.time()

#
# error_free = True
# if error_free:
#     noise_level = 0
#
# # 1. sample LDPC code word
# H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True) #
# # print(H.shape)
# H_diag = diag_format(H, databit_num)
#
# H_permute = np.random.permutation(H)
#
# save_image_data(H, "H_original")
# save_image_data(H_permute, "QCLDPC_permuted")
# save_image_data(H_diag, "QCLDPC_diag_format")


###### parallel execution numba test #####
# @njit(parallel=True)
# def test_numba():
#     for t in prange(10):
#         hash_vec = np.random.randint(0, 2, size=10)  # random hash
#         print(hash_vec)
#
#
# test_numba()
