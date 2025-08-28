from prettyprinter import *
from ECO import *
from ECO_original import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from formatter import *
from variables import *
from dubiner_sparsifyer import *
from data_saver import *
from bit_flip_decoder import ldpc_bitflip_majority_decode
from bit_flip_decoder_parallel import ldpc_bitflip_majority_decode_numba
from bit_flip_decoder_sequential import ldpc_bitflip_seqdecode_numba


import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''

# sample LDPC code word

H, A = sample_LDPC(codeword_len,databit_num,density = density,pooling_factor=pooling_factor,noise_level=noise_level,save_noise_free=True)
A_error_free = read_matrix('error_free_codeword')

total_codewords = pooling_factor * codeword_len
correct_codewords = compare_matrix(A_error_free, A)
# print("Codeword generated: noise is added, so correct  prior codeword number is as follows")
print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)

    print("Code word matrix: ")
    print_arr(A)
else:
    print("Code word generated", end=' - ')

print(" %s seconds" % round(time.time() - start_time,3))

decoding_codeword_matrix = np.copy(A)

decode_num = 3 # do 3 times
for i in range(decode_num):
    print()
    print('#'*30)
    print("Iterative Decoding Loop ", i+1)
    # 1. apply ECO to A(code word matrix) and get Q, R
    Q_aux = np.identity(codeword_len, dtype=np.uint8)

    R,Q = ECO(decoding_codeword_matrix,Q_aux,BGCE=BGCE, debug = True)#ECO_original(A,Q_aux,BGCE=BGCE)
    if not LARGE_CODE:
        print("Result:")
        print_arr(R)
        print("ECO matrix: ")
        print_arr(Q)
    else:
        print("ECO complete",end=' - ')

    print(" %s seconds" % round(time.time() - start_time,3))

    # 2. search for sparse columns in R and get columns in Q of the corresponding indices
    idx = get_sparse_column_idx(R,threshold)

    # 3. after collecting n-k such columns of Q, transpose it to form H
    H_recovered = get_sparse_columns(Q,idx).T

    # 4. extract n-k dual vectors if more than one vector is considered to be sparse
    H_extracted = reliability_extraction(H_recovered, A,parity_num)

    # 5. Sparsify/Format H matrix -> see if it has diagonal or bi-diagonal format
    H_formatted = diag_format(H_extracted, databit_num)
    # H_formatted = H_extracted # formatting (sparsifying 대체) 을 하지 않으면 아예 찾지 못하는 경우도 있음!
    # H_formatted = sparsify(H_extracted, codeword_len*pooling_factor, codeword_len) # doesnt work since ms > ns (it should be opposite)

    # print("Saving recovered H matrix")
    # save_H_matrix(H_formatted)
    # save_image_data(H_formatted)

    decoded_codeword_matrix, ok, _, _, _ = ldpc_bitflip_seqdecode_numba(H_formatted, A, max_iter=50)
    print("Decoding complete",end=' - ')
    print(" %s seconds" % round(time.time() - start_time,3))
    save_matrix(decoded_codeword_matrix, filename='decoded_codeword')
    save_matrix(H, filename='H')

    correct_codewords = compare_matrix(A_error_free, decoded_codeword_matrix)
    print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))


    if not LARGE_CODE:
        print("recovered H matrix row space")
        print_arr(H_extracted)
        print("formatted H")
        print_arr(H_formatted)

    print("Success?: ", check_success(H,H_formatted))
    print("Total elapsed time: %s seconds" % round(time.time() - start_time,3))
    print()

    decoding_codeword_matrix = np.copy(decoded_codeword_matrix)


# 와 7비트의 오류만 있어도 30개가 안찾아지네. 이게 700개가 되어도 안찾아짐. 뭔가 문제가 있는 모양.


















