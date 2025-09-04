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


import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
Progressive reconstruction of QC LDPC matrix in a noisy channel

더 개선할 점
Do a sanity check? 
=> Check amount of error a dual vector detects from the codeword 
=> if too many, it means it is a wrong dual vector(even when there is noise), so reject
'''
# target H matrix
H_final = None
error_free = True
get_all_block_shifts = False

if error_free:
    noise_level = 0

# 1. sample LDPC code word
H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True) #
# print(H.shape)
H_diag = diag_format(H, databit_num)
generator = get_generator(H,databit_num)
# print(generator.shape)

A = get_codewords(generator,codeword_len, databit_num,pooling_factor = pooling_factor,noise_level = noise_level,save_noise_free=True)
A_error_free = read_matrix('error_free_codeword')
save_matrix(A, filename='noisy_codeword')

# print("Base size (mb x nb):", B.shape)
# print("Lifting Z:", Z)
# print("Full H shape:", (mb * Z, nb * Z))
save_image_data(H, filename="n_{}_k_{}".format(codeword_len, databit_num))

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)
    print("Code word matrix: ")
    print_arr(A)
else:
    print("Code word generated")
total_codewords = pooling_factor * codeword_len
correct_codewords = compare_matrix(A_error_free, A)
print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))


print("Elapsed time: %s seconds" % round(time.time() - start_time,3))
#############################################################################################
# Iterative decoding

decoding_codeword_matrix = np.copy(A)

decode_num = 20 # loop N times
for i in range(decode_num):
    print()
    print('#'*30)
    print("Iterative Decoding Loop ", i+1)

    # 2. submatrix sampling : progressive reconstruction of ldpc code 논문의 fig2 참조
    # note: for better performance, ms < ns
    col_factor = 0.8#0.5  # 0.5
    row_factor = 0.8#0.5  # 0.3
    ns = round(col_factor * codeword_len)  # number of sampled col
    ms = round(row_factor * ns)  # number of sampled row

    row_indices, col_indices = sample_row_col_indices(A, ms, ns)
    if error_free:
        sub = sample_submatrix(A, row_indices, col_indices)
    else:
        sub = sample_submatrix(decoding_codeword_matrix, row_indices, col_indices)
    # print(col_indices)
    # print_arr(sub)

    # 3. gaussian elimination 으로 복원하는 방법을 사용 - 이거 말고 (검증된) ECO 방식을 써볼까?
    Gs = gf2elim(sub)
    P = Gs[:, ms:]
    P_t = np.transpose(P)
    Hs = np.concatenate((P_t, np.identity(ns - ms, dtype=np.uint8)), axis=1)
    # print("Hs")
    # print_arr(Hs)
    # print("SVR complete")

    # 4.dubiner sparsification
    Hr = sparsify(Hs, ms, ns, column_swap=False)
    if Hr.shape[0]==0: # no vector found
        print("No sparse vector found")
        continue
    mr, nr = Hr.shape

    # 5. Hs를 다시 H의 규격에 맞게(길이 n) 0을 추가해서 늘려줌
    H_recovered = np.zeros((mr, codeword_len),
                           dtype=np.uint8)  # H_recovered = np.zeros((ns-ms,codeword_len), dtype=np.uint8)
    H_recovered[:, col_indices] = Hr

    # 6. get all the block shifts of blocks
    if get_all_block_shifts:
        for dual_vector in H_recovered:
            shifts = qc_global_cyclic_shifts_numba(dual_vector, Z)  # shift해서 블럭 개수 늘리기 (block size is given, Z)
            if H_final is None:
                H_final = np.array(shifts)
            else:
                H_final = np.concatenate((H_final, shifts), axis=0)
    else:
        if H_final is None:
            H_final = H_recovered
        else:
            H_final = np.concatenate((H_final, H_recovered), axis=0)

    if not error_free:
        # 7. decoding using hard decision bit flip
        decoded_codeword_matrix, ok, _, _, _ = ldpc_bitflip_seqdecode_numba(H_final, A, max_iter=50)
        print("Decoding complete", end=' - ')
        print(" %s seconds" % round(time.time() - start_time, 3))
        # save_matrix(decoded_codeword_matrix, filename='decoded_codeword')
        correct_codewords = compare_matrix(A_error_free, decoded_codeword_matrix)
        print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))
        decoding_codeword_matrix = np.copy(decoded_codeword_matrix)

    if not LARGE_CODE:
        print("formatted H")
        print_arr(H_final)

    print("Success?: ", check_success(H, H_final))
    print("Total elapsed time: %s seconds" % round(time.time() - start_time, 3))

save_image_data(H_final, "recovered_qc")
# Error exaggeration part
# A = read_matrix('decoded_codeword')
# B = read_matrix('error_free_codeword')
# diff = A^B
# save_error_image(A, diff, mode = 'blob')
###############################################################################

this_time = time.time()
print("Success?: ", check_success(H,H_final))



