'''
General sparsifier
Dubiner sparsifier is used when sampled codewords are less than codeword length
Hence we invented sparsifier that works when there are more codewords used than its length

'''

import numpy as np
import numba
from formatter import diag_format
from variables import density, databit_num, codeword_len
import random
from numba import njit, prange

########## parallel version ###########
'''
Hs: ms x ns matrix
where 
ns is number of sampled columns, which is generally smaller than n (codeword length)
ms is number of recovered parity check vectors 

'''
@njit(parallel=True)
def general_sparsifier(Hs, databits, w_th = 14, use_row_combinations = False):
    ms,ns = Hs.shape
    ds = round(databits/2) # hashing 강도
    Niter = 100

    max_candidates = Niter * 100  # rough upper bound (need tuning)
    Hr_tmp = np.zeros((max_candidates, ns), dtype=np.uint8)
    Hr_count = 0

    # if a vector is sparse enough from the beginning, take it
    for i in range(ms): # for each row in Hs
        s = Hs[i]
        s_test = s.astype(np.uint64)
        if 0 < sum(s_test) <= w_th:
            if Hr_count < max_candidates: # append sparse vectors
                Hr_tmp[Hr_count] = s
                Hr_count += 1

    if use_row_combinations:
        for t in prange(Niter):
            # get filter (to get similar vectors in one bucket)
            hash_vec = np.random.randint(0, 2, size=databits)  # random hash
            hash_idx = np.arange(databits)
            np.random.shuffle(hash_idx)
            hash_idx = hash_idx[:ds]

            bucket = []
            for i in range(ms):
                h = Hs[i]
                databit_h = h[:databits]
                match = True
                for idx in hash_idx: # hash index에 대해서만 hash랑 값이 같은지 체크하는듯
                    if databit_h[idx] != hash_vec[idx]:
                        match = False
                        break
                if match:
                    bucket.append(h)

            bn = len(bucket)
            for i in range(bn - 1):
                for j in range(i + 1, bn):
                    s = bucket[i] ^ bucket[j]
                    s_test = s.astype(np.uint64)
                    weight = np.sum(s_test)
                    if 0 < weight <= w_th:
                        if Hr_count < max_candidates:
                            Hr_tmp[Hr_count] = s
                            Hr_count += 1

    return Hr_tmp[:Hr_count]









