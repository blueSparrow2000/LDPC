import numpy as np

def gf2_row_reduce(mat):
    """Row-reduce a binary matrix over GF(2)."""
    mat = mat.copy() % 2
    rows, cols = mat.shape
    pivot_row = 0
    for col in range(cols):
        # Find pivot
        pivot = np.where(mat[pivot_row:, col] == 1)[0]
        if pivot.size == 0:
            continue
        pivot += pivot_row
        # Swap
        if pivot[0] != pivot_row:
            mat[[pivot_row, pivot[0]]] = mat[[pivot[0], pivot_row]]
        # Eliminate other rows
        for r in range(rows):
            if r != pivot_row and mat[r, col] == 1:
                mat[r] ^= mat[pivot_row]
        pivot_row += 1
        if pivot_row >= rows:
            break
    return mat

def gf2_nullspace(mat):
    """Compute nullspace basis of a binary matrix over GF(2)."""
    rows, cols = mat.shape
    # Row reduce
    rref = gf2_row_reduce(mat)
    pivots = []
    free_vars = []
    for i in range(rows):
        pivot_cols = np.where(rref[i]==1)[0]
        if pivot_cols.size > 0:
            pivots.append(pivot_cols[0])
    pivots = set(pivots)
    free_vars = [j for j in range(cols) if j not in pivots]

    basis = []
    for free in free_vars:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[free] = 1
        for i in range(rows):
            pivot_cols = np.where(rref[i]==1)[0]
            if pivot_cols.size == 0:
                continue
            pivot = pivot_cols[0]
            if rref[i, free] == 1:
                vec[pivot] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8)

def recover_H(codewords):
    """
    Recover candidate parity-check matrix H given valid codewords.
    codewords: ndarray (#samples x N), each row a codeword
    """
    # The codewords matrix C, H * c^T = 0 => H * C^T = 0
    # So nullspace of C is the rowspace of H
    C = np.array(codewords, dtype=np.uint8)
    null_basis = gf2_nullspace(C)
    H = null_basis  # Each row is a parity-check equation
    return H
