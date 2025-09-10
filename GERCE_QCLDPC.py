from prettyprinter import *
from verifier import *
from variables import *
from GERCE import GERCE
from QCLDPC_sampler import *
from submatrix_sampling import *
from block_recover import *
from verifier import *
from formatter import *
from data_saver import *
from csv_writter import *
from bit_flip_decoder_sequential import ldpc_bitflip_seqdecode_numba
from sparsifier import *
from extracter import *

import time
start_time = time.time()

'''
Implementation of LDPC PCM recovering method of paper:
A fast reconstruction of the parity check matrices of LDPC codes in a noisy environment (2021)
'''

# meta info
skip_generation = True
error_free = False #False
get_all_block_shifts = True #True
use_gauss_elim = False
recovered_data_mode = 'w' # 'w': write mode / 'a': append mode, add recovered PCM into existing H file

if error_free:
    noise_level = 0

if skip_generation:
    H = read_matrix("H_true")
    A = read_matrix("noisy_codeword")
    A_error_free = read_matrix('error_free_codeword')
else:
    # 1. sample LDPC code word
    H, B, S = generate_nand_qc_ldpc(mb=mb, nb=nb, Z=Z, target_rate=target_rate, rng=1234, sparse=True) #
    save_matrix(H, 'H_true')
    save_image_data(H, filename="n_{}_k_{}".format(codeword_len, databit_num))
    H_diag = diag_format(H, databit_num)
    generator = get_generator(H,databit_num)
    A = get_codewords(generator,codeword_len, databit_num,pooling_factor = pooling_factor,noise_level = noise_level,save_noise_free=True)
    A_error_free = read_matrix('error_free_codeword')
    save_matrix(A, filename='noisy_codeword')

total_codewords = pooling_factor * codeword_len
correct_codewords = compare_matrix(A_error_free, A)
print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))
#############################################################################################

kappa = 0 # num_dual_vectors
v = round(codeword_len*0.5) # codeword_len #  # col_extract_num
vmin = 0


Nmin = 50 #100 # parameter determined by v, n and BER (more than 100)
GERCE_iter = 1 # number permutation iteration in GERCE - pass in 1 => dont permute
total_iteration = 10 # full iteration number - decoding is done this amount

mainIter = 0

np.random.seed(2) # randomize

H_final = None  # currently found dual vectors
H_final_blocks = None # recovered 'blocks', generated from H_final
H_recovered = None
decoding_codeword_matrix = np.copy(A)

while mainIter<total_iteration:
    # this vector is used for only block shift purposes
    newerly_recovered_vec = None # keep track of number of recovered vectors in a total loop

    for i in range(Nmin):
        if (i+1)%10 == 0: # show every 10th iteration count
            print("{}th iteration".format(i+1))
            print("Current vectors: ", kappa)
        # 1. Sample col indices
        col_indices = sample_col_indices(decoding_codeword_matrix, v)
        Mv = decoding_codeword_matrix[:, col_indices]

        if use_gauss_elim:
            # gauss elim method
            Gs = gf2elim(Mv)
            mv,_ = Gs.shape
            # zero pad Hv to form PCM in dim n space
            G_recovered = np.zeros((mv, codeword_len),
                             dtype=np.uint8)
            G_recovered[:, col_indices] = Gs

            # 아래도 formatting이후에 해야하는 작업임!
            P = G_recovered[:, databit_num:]
            P_t = np.transpose(P)
            H_pad = np.concatenate((P_t, np.identity(parity_num, dtype=np.uint8)), axis=1)
            mv = parity_num

        else:
            # 2. GERCE
            Hv = GERCE(Mv,Niter= GERCE_iter, QCLDPC=True, use_ECO=True)

            # 3. sparsification
            H_sparse = general_sparsifier(Hv,databit_num, w_th= 13+2,use_row_combinations=False)
            mv, nv = H_sparse.shape

            # 4. zero pad Hv to form PCM in dim n space
            H_pad = np.zeros((mv, codeword_len), dtype=np.uint8)
            H_pad[:, col_indices] = H_sparse

        # 5. remove vectors that does not satisfy block constraint(weight 1 in a row per block etc.)
        H_candidate = get_block_candidates(H_pad, codeword_len, Z)
        H_candidate = np.array(H_candidate)

        # 6. check L.I.
        for j in range(mv): # for each recovered dual vectors, check Linear Independency with previously found dual vectors
            h = H_candidate[j] # a dual vector
            if H_recovered is None:
                H_recovered = np.array([h]) # initial vector
                kappa += 1

                newerly_recovered_vec = np.array([h])  # initial vector
                continue
            Htemp = np.append(H_recovered, [h], axis = 0)
            if np.linalg.matrix_rank(Htemp) > kappa: # increase rank
                H_recovered = np.append(H_recovered, [h], axis=0) #add h into H
                kappa += 1

                if newerly_recovered_vec is None:
                    newerly_recovered_vec = np.array([h])  # initial vector
                else:
                    newerly_recovered_vec = np.append(newerly_recovered_vec, [h], axis=0)  # add h into H

    # 7. recover blocks
    if get_all_block_shifts:
        if not (newerly_recovered_vec is None): # if H_recovered is not None, and there is newerly found vectors
            for dual_vector in newerly_recovered_vec:  # for dual_vector in H_candidate: - sample L.I ones
                shifts = qc_global_cyclic_shifts_numba(dual_vector, Z)  # shift해서 블럭 개수 늘리기 (block size is given, Z)
                if H_final is None:
                    H_final = np.array(shifts)
                else:
                    H_final = np.concatenate((H_final, shifts), axis=0)
        else:
            print("H_recovered is None")
    else:
        H_final = H_recovered

    # 8. decoding using hard decision bit flip
    if not error_free and not (newerly_recovered_vec is None): # 새로 발견되는게 있을때만
        decoded_codeword_matrix, ok, _, _, _ = ldpc_bitflip_seqdecode_numba(H_final, A, max_iter=50)
        print("Decoding complete", end=' - ')
        print(" %s seconds" % round(time.time() - start_time, 3))
        # save_matrix(decoded_codeword_matrix, filename='decoded_codeword')
        correct_codewords = compare_matrix(A_error_free, decoded_codeword_matrix)
        print("Correct / Total = {} / {}".format(correct_codewords, total_codewords))
        decoding_codeword_matrix = np.copy(decoded_codeword_matrix)

    mainIter += 1

this_time = time.time()
print("Main loop done - ", end = '')
print("Elapsed time: %s seconds" % round(this_time - start_time,3))

# reliability extraction
# H_final = reliability_extraction(H_final, A, round(parity_num/Z)+2) # we need block seeds -> get as much as we can? parity_num amount?
# this_time = time.time()
# print("reliability extraction done - ", end = '')
# print("Elapsed time: %s seconds" % round(this_time - start_time,3))
# print(H_final.shape)

print()
print("Success?: ", check_success(H,H_final))


try:
    if recovered_data_mode == 'w':
        save_matrix(H_final, 'H_recovered')
        save_image_data(H_final, "recovered_qc")
    elif recovered_data_mode == 'a': # append mode
        # load the matrix and draw a new binary image - used in append mode
        save_matrix(H_final, 'H_recovered',mode='a')
        H_recovered = read_matrix("H_recovered")
        save_image_data(H_recovered, "recovered_qc")

except:
    print("saving img of H failed. Here is what H looks like")
    print(H)


'''
Comment:
희안하게도 이 방법으로 구하면 progressive reconst로 구한 벡터의 여집합이 잘 구해진다... 하
서로 강점인 분야가 다른건가? ECO 기반 방식과 gauss elim기반 방식의 차이점인걸까? 
만약 그렇다면 QC 실험에 gauss elim이 아닌 ECO를 사용해서 한다면 이 실험처럼 마지막 두칸이 빈 블록 뺴고 다 나올지도

=> 그럼 앙상블처럼 gauss elim을 이용하는 방식과 ECO를 쓰는 방식을 모두 써야 한다
'''