'''
Some variables to input
'''
codeword_len = 1000
databit_num = 900
density = 0.15  # number of ones in a P matrix
if codeword_len > 500:
    density = 0.08
elif codeword_len >= 2000:
    density = 0.05

LARGE_CODE = True if codeword_len > 50 else False
parity_num = codeword_len-databit_num
noise_level= 10
pooling_factor = 5
threshold = ((pooling_factor-1)*codeword_len)*0.325  # suggested beta coeff on the paper
BGCE = True # without BGCE, one may get more dual vectors but they are likely to be erronous
if not BGCE: # if GCE, higher the threshold
    threshold *= 2
# print(threshold)