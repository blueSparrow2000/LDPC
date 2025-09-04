import numpy as np
from numba import njit, prange, set_num_threads
from gauss_elim import gf2elim
'''
Formatting is very important! 
This may reduce so much errors
'''


'''
Format given matrix to H matrix form
which has identity matrix at the last n-k columns
'''
def diag_format(H, databit_num):
    p,n = H.shape
    H_before = np.concatenate((H[:, databit_num:],H[:,:databit_num]), axis=1)
    H_gauss = gf2elim(H_before)
    result = np.concatenate(( H_gauss[:, n-databit_num:],H_gauss[:, :n-databit_num]), axis=1)  #
    result = result[~np.all(result == 0, axis=1)] # remove zero rows
    return result

'''
QC LDPC formatting
by GPT
'''
# ---------- Helpers: bit-packing ----------
def pack_bits_matrix(A):
    """
    Pack binary matrix A (shape m x n) into uint64 words (m x nwords).
    A elements must be 0/1 integers.
    """
    A = np.asarray(A, dtype=np.uint8)
    m, n = A.shape
    nwords = (n + 63) // 64
    P = np.zeros((m, nwords), dtype=np.uint64)
    for i in range(m):
        for j in range(n):
            if A[i, j]:
                word = j // 64
                bit = j % 64
                P[i, word] |= (np.uint64(1) << np.uint64(bit))
    return P

def pack_bits_vector(v):
    v = np.asarray(v, dtype=np.uint8)
    n = v.size
    nwords = (n + 63) // 64
    p = np.zeros(nwords, dtype=np.uint64)
    for j in range(n):
        if v[j]:
            word = j // 64
            bit = j % 64
            p[word] |= (np.uint64(1) << np.uint64(bit))
    return p

# ---------- Numba GF(2) elimination on packed rows ----------
@njit
def _find_pivot_row(packed, start_row, col_word_idx, col_bit_mask, nrows, nwords):
    """
    find row r >= start_row with bit (col_word_idx, col_bit_mask) set.
    return r or -1
    """
    for r in range(start_row, nrows):
        if (packed[r, col_word_idx] & col_bit_mask) != 0:
            return r
    return -1

@njit
def gf2_rank_packed(packed, nbits):
    """
    Compute rank of packed binary matrix.
      packed : (r x nwords) uint64
      nbits  : number of columns (bits)
    Gaussian elimination in-place on a copy in Numba.
    Returns rank (int).
    """
    nrows, nwords = packed.shape
    # copy to avoid mutating caller
    M = np.empty((nrows, nwords), dtype=np.uint64)
    for i in range(nrows):
        for j in range(nwords):
            M[i, j] = packed[i, j]

    rank = 0
    # iterate over bit positions 0..nbits-1
    for bit in range(nbits):
        word_idx = bit // 64
        bit_pos = bit % 64
        mask = np.uint64(1) << np.uint64(bit_pos)

        pivot = -1
        # find pivot row
        for r in range(rank, nrows):
            if (M[r, word_idx] & mask) != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        # swap pivot row into 'rank' position
        if pivot != rank:
            for w in range(nwords):
                tmp = M[rank, w]
                M[rank, w] = M[pivot, w]
                M[pivot, w] = tmp
        # eliminate bit from all other rows
        for r in range(nrows):
            if r != rank and (M[r, word_idx] & mask) != 0:
                for w in range(nwords):
                    M[r, w] ^= M[rank, w]
        rank += 1
        if rank >= nrows:
            break
    return rank

@njit
def gf2_rank_packed_with_append(packed, vec_packed, nbits):
    """
    Compute rank after appending vec_packed to packed (like checking if vec is in rowspace):
    Return (rank_original, rank_with_vec).
    """
    nrows, nwords = packed.shape
    # build stacked matrix (nrows+1) x nwords
    S = np.empty((nrows + 1, nwords), dtype=np.uint64)
    for i in range(nrows):
        for j in range(nwords):
            S[i, j] = packed[i, j]
    for j in range(nwords):
        S[nrows, j] = vec_packed[j]

    rank_orig = gf2_rank_packed(packed, nbits)
    rank_with = gf2_rank_packed(S, nbits)
    return rank_orig, rank_with

# ---------- shift within blocks (unpacked) ----------
def shift_blocks_unpacked(vec, Z, s):
    """Shift each block of length Z by s to the right (regular numpy)."""
    N = vec.size
    assert N % Z == 0
    nb = N // Z
    out = np.zeros_like(vec)
    for b in range(nb):
        block = vec[b*Z:(b+1)*Z]
        out[b*Z:(b+1)*Z] = np.roll(block, s)
    return out

# ---------- Numba-parallel check of Z shifts for one candidate ----------
@njit(parallel=True)
def _check_all_shifts_in_rowspace(packed_H, nbits, vec_unpacked, Z, nwords):
    """
    For a candidate base row vec_unpacked (0/1 ndarray length nbits),
    check for s = 0..Z-1 whether shift_blocks(vec, Z, s) is in rowspace.
    Returns a boolean array ok_shifts[s] (True if in rowspace).
    This function is Numba-jitted and parallel over s.
    """
    ok = np.zeros(Z, dtype=np.bool_)
    # pack vec for each shift and test membership
    # we will need a temporary packing array for each shift
    for s in prange(Z):
        # pack shifted vector into uint64 words
        # first compute shifted vector into temporary byte array (we can't call np.roll in njit easily)
        # do block-wise roll manually
        nb = nbits // Z
        # create temporary packed array
        tmp_packed = np.zeros(nwords, dtype=np.uint64)
        for b in range(nb):
            # find positions of ones in this block after shift s
            # for each bit k in block:
            for k in range(Z):
                val = vec_unpacked[b*Z + k]
                if val:
                    newpos = (k + s) % Z
                    global_bit = b*Z + newpos
                    w = global_bit // 64
                    bitpos = global_bit % 64
                    tmp_packed[w] |= (np.uint64(1) << np.uint64(bitpos))
        # now test membership by comparing rank
        rank_orig = gf2_rank_packed(packed_H, nbits)
        # compute rank of stacked (packed_H + tmp_packed)
        # build small stacked matrix
        rrows = packed_H.shape[0]
        S = np.empty((rrows + 1, nwords), dtype=np.uint64)
        for i in range(rrows):
            for j in range(nwords):
                S[i, j] = packed_H[i, j]
        for j in range(nwords):
            S[rrows, j] = tmp_packed[j]
        rank_with = gf2_rank_packed(S, nbits)
        ok[s] = (rank_with == rank_orig)
    return ok

# ---------- Top-level recovery using Numba-accelerated inner ops ----------
def recover_qc_base_from_rowspace_numba(H_raw, Z, try_pairwise_xor=True, max_pair_checks=5000, verbose=False, n_threads=None):
    """
    Same algorithm as previous recover_qc_base_from_rowspace but with numba-accelerated inner ops.
    Note: outer candidate loop remains Python-level greedy; inner membership/shift checks are JIT'd.
    """
    if n_threads is not None:
        set_num_threads(n_threads)

    H_raw = np.asarray(H_raw, dtype=np.uint8)
    m, N = H_raw.shape
    if N % Z != 0:
        raise ValueError("Z must divide N")
    nwords = (N + 63) // 64

    # pack H
    packed_H = pack_bits_matrix(H_raw)  # uint64 array (m x nwords)

    # compute original rank once using jitted function on packed matrix
    rank_orig = gf2_rank_packed(packed_H.astype(np.uint64), N)

    # build candidate pool (rows)
    candidates = [H_raw[i].copy() for i in range(m)]

    # add some pairwise XORs (limited)
    if try_pairwise_xor:
        limit = min(m, 200)
        pair_count = 0
        for i in range(limit):
            for j in range(i+1, limit):
                candidates.append((H_raw[i] ^ H_raw[j]))
                pair_count += 1
                if pair_count >= max_pair_checks:
                    break
            if pair_count >= max_pair_checks:
                break
        if verbose:
            print("Added", pair_count, "pairwise candidates")

    found_bases = []
    found_rows = []

    # helper: compute current rank of found_rows (use packed rank)
    def rank_of_found():
        if len(found_rows) == 0:
            return 0
        P = pack_bits_matrix(np.vstack(found_rows))
        return gf2_rank_packed(P.astype(np.uint64), N)

    # iterate candidates sequentially
    for idx, cand in enumerate(candidates):
        if rank_of_found() >= rank_orig:
            break
        # quick block sparsity check
        nb = N // Z
        ok_block = True
        for b in range(nb):
            ssum = int(cand[b*Z:(b+1)*Z].sum())
            if ssum > 1:
                ok_block = False
                break
        if not ok_block:
            continue

        # call numba-parallel function that checks all shifts for this candidate
        ok_shifts = _check_all_shifts_in_rowspace(packed_H.astype(np.uint64), N, cand.astype(np.uint8), Z, nwords)

        # must ensure every shift is in rowspace
        all_in = True
        for s in range(Z):
            if not ok_shifts[s]:
                all_in = False
                break
        if not all_in:
            continue

        # build the orbit list (Z vectors)
        orbit = [shift_blocks_unpacked(cand, Z, s) for s in range(Z)]

        # ensure independence: rank(found_rows U orbit) == rank(found_rows) + Z
        prev_rank = rank_of_found()
        combined = found_rows + orbit
        P_comb = pack_bits_matrix(np.vstack(combined))
        new_rank = gf2_rank_packed(P_comb.astype(np.uint64), N)
        if new_rank != prev_rank + Z:
            continue

        # accept orbit
        found_rows.extend(orbit)
        found_bases.append(cand.copy())
        if verbose:
            print("Accepted a base from candidate idx", idx)

    final_rank = rank_of_found()
    if final_rank < rank_orig:
        return False, None, found_bases, f"Could not recover full rank ({final_rank} < {rank_orig})"

    # build B base-matrix
    Mb = len(found_bases)
    Nb = N // Z
    B = -np.ones((Mb, Nb), dtype=int)
    for gi, base in enumerate(found_bases):
        for bj in range(Nb):
            block = base[bj*Z:(bj+1)*Z]
            w = int(block.sum())
            if w == 0:
                B[gi, bj] = -1
            elif w == 1:
                B[gi, bj] = int(np.flatnonzero(block)[0])
            else:
                B[gi, bj] = -2

    return True, B, found_bases, "success"



'''
QC base matrix 형태를 H matrix 형태로 바꿔준다 
'''
def expand_qc_base(B, Z):
    """
    Expand a QC base matrix B (shifts) into the full binary parity-check matrix H.

    Parameters
    ----------
    B : (Mb x Nb) ndarray of int
        Base matrix produced by qc_ldpc_format or designed by hand:
          - entries >= 0: shift value (0..Z-1)
          - entry == -1: zero block
          - entry == -2: non-circulant/invalid block -> treated as zero (or you can raise)
    Z : int
        Circulant/block size

    Returns
    -------
    H : (Mb*Z x Nb*Z) ndarray of {0,1}
    """
    B = np.asarray(B, dtype=int)
    Mb, Nb = B.shape
    M, N = Mb * Z, Nb * Z
    H = np.zeros((M, N), dtype=np.int64)

    for bi in range(Mb):
        for bj in range(Nb):
            val = B[bi, bj]
            if val < 0:
                # -1: zero block, -2: invalid; skip (leave zeros)
                continue
            s = int(val) % Z
            # fill block with circulant permutation: row r has a 1 at column (s + r) % Z
            r0 = bi * Z
            c0 = bj * Z
            for r in range(Z):
                H[r0 + r, c0 + ((s + r) % Z)] = 1
    return H

