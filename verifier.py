import numpy as np
from gauss_elim import gf2elim

'''
Compares element wise between two matrix
Return True if identical
'''
def check_success(target_array,my_array):
    if my_array is None: # failed to recover any dual vectors
        print("Failed to recover any dual vector!")
        return False
    tx,ty = target_array.shape
    mx,my = my_array.shape

    correct_guess = 0
    for i in range(tx):
        if (my_array == target_array[i]).all(axis = 1).any():
            correct_guess += 1

    if tx > mx: # too small!
        print("Dual vectors missing: ", tx-mx)
        print("correct guess: {} / {}".format(correct_guess,tx))
        return False
    if tx == mx and (target_array == my_array).all():
        # perfectly correct
        return True

    if correct_guess== tx: # found all but too many
        print("Found all {} parity check vectors".format(tx))
        print("Total dual vectors recovered: {}".format(mx))
        return False

    print("Total dual vectors recovered: {}".format(mx))
    print("correct guess: {} / {}".format(correct_guess,tx))
    return False

