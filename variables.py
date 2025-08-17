'''
Some variables to input
'''
import numpy as np
codeword_len = 1000
databit_num = 800
density = 0.15  # number of ones in a P matrix
if codeword_len > 500:
    density = 0.08
elif codeword_len >= 2000:
    density = 0.05

LARGE_CODE = True if codeword_len > 50 else False
parity_num = codeword_len-databit_num
noise_level= 10
pooling_factor = 1.5
BGCE = True # without BGCE, one may get more dual vectors but they are likely to be erronous
threshold = round(((pooling_factor-1)*codeword_len)*0.325)  # suggested beta coeff on the paper
if not BGCE: # if GCE, higher the threshold
    threshold = round((pooling_factor*codeword_len)*0.325) # codeword_amount/2 * beta_opt (0.65, based on experience)

NOISE_PROB = 0.0001

np.random.seed(seed=0) # 6
# print(threshold)

PRINT_VAR_SETTING = True
# print variable settings
if PRINT_VAR_SETTING:
    print('#' * 5, "variable settings", '#' * 5)
    print("(n, k) = ({}, {})".format(codeword_len, databit_num))
    print("Number of code words: ", pooling_factor*codeword_len)
    print("Threshold: ", threshold)
    print('#' * 30)