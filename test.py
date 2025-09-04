import numpy as np
from data_saver import *
from csv_writter import *
'''
# difference comparison
A = read_matrix('decoded_codeword')
B = read_matrix('error_free_codeword')
C = read_matrix('noisy_codeword')
# diff_origin = B^C
# save_image_data(diff_origin, filename="noise_map")
diff = A^B

def remove_error(A,diff):
    delete_indices = []
    error_by_row = np.sum(diff, axis=1)
    l = len(error_by_row)
    for i in range(l):
        if error_by_row[i]>0: # more than x errors
            delete_indices.append(i) # get the index of the row

    for i in range(len(delete_indices)):
        el = delete_indices.pop()
        A = np.delete(A, (el), axis=0) # delete row

    # saved error-free version
    save_matrix(A, filename='decoded_codeword')



save_error_image(A, diff, mode = 'blob')

# remove_error(A,diff)
'''
H_recovered = np.array([[1,0,0],[0,1,0]])
H = None
for dual_vector in H_recovered:
    shifts = np.array([[1,0,0],[0,1,0]]) # shift해서 블럭 개수 늘리기

    if H is None:
        H = np.array(shifts)
    else:
        H = np.concatenate((H,shifts), axis = 0)

print(H)






