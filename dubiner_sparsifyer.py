import numpy as np
import numba
from formatter import diag_format
from variables import density, databit_num, codeword_len
import random

'''
This is an implementation of algorithm 1 in paper "Progressive reconstruction of qc ldpc ..."

Hyper parameter (not shown in the paper)
W_th : weight threshold - for QC LDPC code with unkown w, how can we determine w value?
d_s  : how strictly to bucket the codes
N_iter (can be approximately calculated)

문제 => 답이 아닌 vector가 많이 찾아진다. 
column swap을 해도 되는것인가!
'''
 #
def get_w_th():
    global density, databit_num
    return (density*databit_num)  # weight threshold

def get_N_iter():
    global databit_num, codeword_len
    return 1000#codeword_len-databit_num # approximate to n-k

def sparsify(Hs, ms,ns, column_swap = True):
    ds = round(ms/2) # hashing 강도
    w_th= 12 + 2#get_w_th()
    Niter = get_N_iter()
    # if not column_swap:
    #     Niter = 1

    Hr = []

    # if a vector is sparse enough from the beginning, take it
    for i in range(ns-ms): # for each row in Hs
        s = Hs[i]
        if 0<sum(s) <= w_th:
            Hr.append(s)  # append sparse vectors

    for t in range(Niter):
        bucket = []
        hash = np.random.choice([0, 1], size=(ms,), p=[1./2, 1./2])

        hash_survive_idx = np.array([True] * ds + [False] * (ms-ds))
        hash_indicies = np.arange(ms)

        hash_idx = hash_indicies[np.random.permutation(hash_survive_idx)]
        for i in range(ns-ms):
            h = Hs[i] # get i th row of Hs
            databit_h = h[:ms]
            # print(databit_h)
            # print(hash)

            if (databit_h[hash_idx] == hash[hash_idx]).all(): # check hash of databits
                bucket.append(h)

        bn = len(bucket)
        # print("collisions: ",bn)
        for i in range(bn-1):
            for j in range(i+1, bn):
                s = bucket[i] ^ bucket[j]
                if 0<sum(s) <= w_th:
                    Hr.append(s) # append sparse vectors

        if column_swap:
            j = random.randint(0,ms-1)
            k = random.randint(ms,ns-1)
            # swap col of Hc
            temp = np.copy(Hs[:,j])
            Hs[:, j] = Hs[:, k]
            Hs[:, k] = temp
            # format Hs into H matrix format
            Hs = diag_format(Hs, ms)
    return np.array(Hr)







