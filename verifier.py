import numpy as np
from gauss_elim import gf2elim

'''
Compares element wise between two matrix
Return True if identical
'''
def check_success(target_array,my_array):
    tx,ty = target_array.shape
    mx,my = my_array.shape
    if tx > mx: # too small!
        print("Dual vectors missing: ", tx-mx)
        return False
    return (target_array == my_array).all()

'''
Format given matrix to H matrix form 
which has identity matrix at the last n-k columns
'''
def diag_format(H, databit_num):
    p,n = H.shape
    H_before = np.concatenate((H[:, databit_num:],H[:,:databit_num]), axis=1)
    H_gauss = gf2elim(H_before)
    result = np.concatenate(( H_gauss[:, n-databit_num:],H_gauss[:, :n-databit_num]), axis=1)  #
    return result