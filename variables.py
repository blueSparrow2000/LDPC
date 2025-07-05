'''
Some variables to input
'''
codeword_len = 10
LARGE_CODE = True if codeword_len > 50 else False
databit_num = 6
parity_num = codeword_len-databit_num
noise_level=2
pooling_factor = 2
threshold = ((pooling_factor-1)*codeword_len)*0.325  # suggested beta coeff on the paper
# print(threshold)