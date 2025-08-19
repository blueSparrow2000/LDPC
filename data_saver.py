'''
Save the data into .txt file
'''
import numpy as np
from variables import codeword_len, databit_num

def save_data(filename, data):
    with open('H_saves/%s.txt'%filename, 'w', newline='') as f:
        f.write(data)


def save_H_matrix(H):
    m,n = H.shape
    data = ""
    for i in range(m):
        for j in range(n):
            data += "%d"%(H[i,j])
        data += "\n"
    filename = "n_{}_k_{}".format(codeword_len, databit_num)
    save_data(filename, data)

# convert data into H matrix
def read_H_matrix(filename):
    H = []
    with open('H_saves/%s.txt'%filename, 'r', newline='') as f:
        lines = f.readlines()
        for line in lines:
            row = []
            line = line.strip()  # delete newline characters
            for bit in line:
                row.append(int(bit))
            H.append(row)

    return np.array(H)


# Hmat_saved = read_H_matrix("n_10_k_8")
# print(Hmat_saved)

