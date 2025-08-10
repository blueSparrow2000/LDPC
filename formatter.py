import numpy as np
from gauss_elim import gf2elim
'''
Formatting is very important! 
This may reduce so much errors
'''


'''
Format given matrix to H matrix form
which has identity matrix at the last n-k columns
'''
def diag_format(H, databit_num):
    p,n = H.shape
    H_before = np.concatenate((H[:, databit_num:],H[:,:databit_num]), axis=1)
    H_gauss = gf2elim(H_before)
    result = np.concatenate(( H_gauss[:, n-databit_num:],H_gauss[:, :n-databit_num]), axis=1)  #
    result = result[~np.all(result == 0, axis=1)] # remove zero rows
    return result


def qc_format(H, databit_num):
    return
