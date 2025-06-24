import numpy as np

''' 
threshold: number of 1's that is acceptable - rule of thumb to 0.65*(M-n)/2 considering noise rate


'''
def get_sparse_column_idx(A, threshold = 0):
    # for each column, count number of ones
    counts = A.sum(0)
    sparse_check = np.array([True if x <= threshold else False for x in counts])
    # print(sparse_check)
    return sparse_check

def get_sparse_columns(Q,idx):
    idx = idx.reshape(Q.shape[0])
    return Q[:,idx==True]


# A = np.array([[1,0,1],[0,0,1],[0,0,0]])
# get_sparse_column(A)










