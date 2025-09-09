'''
Block recovering function
Input: a parity check (PCV), block size p
Output: 'p' parity check vectors related to input PCV
which is a right shifts of such row
'''
import numpy as np
from numba import njit, prange

# p: block size

@njit(parallel=True)
def qc_global_cyclic_shifts_numba(vec, p):
    """
    Parallelized version with Numba.
    Generate all global cyclic shifts of a QC-LDPC parity check vector,
    where each block (size p) is shifted independently mod p,
    but all blocks are shifted simultaneously.

    Parameters
    ----------
    vec : 1D numpy array of {0,1}, length n
    p : int, block size

    Returns
    -------
    shifts : 2D numpy array of shape (p, n)
             shifts[s, :] = vector after shift by s
    """
    n = vec.shape[0] #n = len(vec)
    if n % p != 0:
        raise ValueError("Vector length must be multiple of block size p")
    num_blocks = n // p

    shifts = np.zeros((p, n), dtype=np.uint8)

    for s in prange(p):  # shift amount
        for b in range(num_blocks):  # block index
            for j in range(p):  # inside block
                src_idx = b * p + j
                dst_idx = b * p + ((j + s) % p)
                shifts[s, dst_idx] = vec[src_idx]

    return shifts

def get_block_candidates(H_recovered, codeword_len, Z):
    # check block distances => there should be only one 1 per row in a block -> hence there should not be 1's in the same block range
    H_candidate = []
    if H_recovered is None:
        # print("No vector is recovered!")
        return H_candidate

    for i in range(len(H_recovered)):
        add_vector = True
        h = H_recovered[i]
        # check whether there exists multiple 1's in a block, if so, delete it
        num_blocks_per_row = int(codeword_len/Z)
        for p in range(num_blocks_per_row):
            row_block = h[p*Z:(p+1)*Z]
            if sum(row_block) > 1: # invalid vector!
                add_vector = False # remove such vector
        if add_vector:
            H_candidate.append(h)

    return H_candidate


# small test example
if __name__ == "__main__1":
    p = 4
    n = 4 * p
    vec = np.zeros(n, dtype=np.uint8)
    vec[2] = 1   # block 0, pos=0
    vec[4] = 1   # block 1, pos=0
    vec[9] = 1   # block 2, pos=1

    shifts = qc_global_cyclic_shifts_numba(vec, p)

    print("Original:", vec)
    for i in range(p):
        print(f"Shift {i}:", shifts[i])
































