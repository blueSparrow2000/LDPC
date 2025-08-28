import numpy as np


def ldpc_encode(H, info_bits):
    """
    Systematic LDPC encoder for QC-LDPC with double-diagonal parity structure.

    Parameters
    ----------
    H : ndarray (M x N)
        Parity-check matrix (0/1 numpy array).
        Assumed partitioned as [H_u | H_p], where H_p is lower-triangular block-diagonal.
    info_bits : ndarray (K,)
        Information bits (0/1 array).

    Returns
    -------
    codeword : ndarray (N,)
        Encoded LDPC codeword [info_bits | parity_bits].
    """
    M, N = H.shape
    K = N - M  # number of information bits
    assert len(info_bits) == K, f"Expected {K} info bits, got {len(info_bits)}"

    # Partition H
    H_u = H[:, :K]
    H_p = H[:, K:]

    # Compute syndrome contribution from info bits
    s = (H_u @ info_bits) % 2  # length M

    # Solve for parity bits using back substitution
    p = np.zeros(M, dtype=np.uint8)
    for i in range(M):
        # Equation: sum_j H_p[i,j] * p[j] = s[i] (mod 2)
        # Since H_p has identity (or shifted identity) on the diagonal, we can solve sequentially
        lhs = 0
        for j in range(i):  # already solved parities
            if H_p[i, j] == 1:
                lhs ^= p[j]
        # diagonal is always 1 (assumption: double diagonal structure)
        p[i] = (s[i] ^ lhs) & 1

    # Build full codeword
    codeword = np.concatenate([info_bits, p])
    return codeword


def encode_multiple(H, num_codewords):
    """
    Generate and encode random info words into LDPC codewords.

    Parameters
    ----------
    H : ndarray (M x N)
        QC-LDPC parity check matrix.
    num_codewords : int
        How many codewords to generate.

    Returns
    -------
    codewords : ndarray (num_codewords x N)
        Encoded codeword matrix.
    """
    M, N = H.shape
    K = N - M
    codewords = np.zeros((num_codewords, N), dtype=np.uint8)

    for i in range(num_codewords):
        info_bits = np.random.randint(0, 2, size=K, dtype=np.uint8)
        codewords[i] = ldpc_encode(H, info_bits)

    return codewords


##

# Example: suppose H is (6 x 12), rate 1/2 code
H = np.array([
    [1,0,1,0,1,0, 1,0,0,0,0,0],
    [0,1,0,1,0,1, 1,1,0,0,0,0],
    [1,1,0,0,1,0, 0,1,1,0,0,0],
    [0,0,1,1,0,1, 0,0,1,1,0,0],
    [1,0,0,0,1,1, 0,0,0,1,1,0],
    [0,1,1,0,0,0, 0,0,0,0,1,1]
], dtype=np.uint8)

codewords = encode_multiple(H, 5)
print(codewords)