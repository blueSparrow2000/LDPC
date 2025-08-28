import numpy as np
from LDPC_sampler import *
from variables import *
import time
from verifier import compare_matrix
from numba import njit, prange

# 큰 차이는 아니지만 이게 조금 더 빠르다. 1.3배? 정도 더 빠른듯


@njit(parallel=True)
def ldpc_bitflip_majority_decode_numba(H, codewords, max_iter=100, weight_tolerance=2):
    """
    Parallel Gallager-B majority bit-flip LDPC decoder (Numba, prange).
    Same semantics and rollback rules as your original function, but:
      - deterministic tie-break (first candidate) for Numba compatibility.
      - parallelized across codewords.

    Inputs
    ------
    H : (m x n) int64  (0/1)
    codewords : (B x n) int64 (0/1)
    max_iter : int
    weight_tolerance : int  (tolerance for weight increase rollback)

    Returns
    -------
    decoded : (B x n) int64
    success : (B,) bool
    iters : (B,) int64
    syndromes : (B x m) int64
    flips : (B,) int64
    """
    # --- sanitize shapes (no exceptions in njit/paral; assume valid shapes) ---
    C = codewords.copy()
    B, n = C.shape
    m, nH = H.shape
    # prepare outputs
    decoded = np.empty((B, n), dtype=np.uint8)
    success = np.zeros(B, dtype=np.bool_)
    iters = np.zeros(B, dtype=np.uint8)
    syndromes = np.zeros((B, m), dtype=np.uint8)
    flips = np.zeros(B, dtype=np.uint8)

    # Precompute deg_v and tau
    deg_v = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        sdeg = 0
        for i in range(m):
            if H[i, j] != 0:
                sdeg += 1
        deg_v[j] = sdeg
    tau = deg_v // 2

    # Parallel loop over codewords
    for b in prange(B):
        w = C[b].copy()
        orig_w = C[b].copy()
        last_w = np.empty(n, dtype=np.uint8)
        for jj in range(n):
            last_w[jj] = -1
        total_flips = 0
        finished = False

        for it in range(1, max_iter + 1):
            # compute syndrome s = H * w (mod 2) via XOR
            s = np.zeros(m, dtype=np.uint8)
            nonzero = 0
            for i in range(m):
                acc = 0
                for j in range(n):
                    if H[i, j] != 0:
                        acc ^= (w[j] & 1)
                s[i] = acc
                if acc != 0:
                    nonzero += 1

            if nonzero == 0:
                # success. apply rollback check on Hamming weight increase
                if np.sum(w) > np.sum(orig_w) + weight_tolerance:
                    decoded[b, :] = orig_w
                    success[b] = False
                else:
                    decoded[b, :] = w
                    success[b] = True
                iters[b] = it - 1
                syndromes[b, :] = s
                flips[b] = total_flips
                finished = True
                break

            # u = H^T @ s  (counts of unsatisfied checks per variable)
            u = np.zeros(n, dtype=np.uint8)
            for j in range(n):
                acc = 0
                if deg_v[j] > 0:
                    for i in range(m):
                        if H[i, j] != 0 and s[i] != 0:
                            acc += 1
                u[j] = acc

            # majority flip mask
            any_flip = False
            flip_mask = np.zeros(n, dtype=np.uint8)
            for j in range(n):
                if u[j] > tau[j] and deg_v[j] > 0:
                    flip_mask[j] = 1
                    any_flip = True

            if not any_flip:
                # pick first variable with maximum u (deterministic tie-break)
                umax = 0
                for j in range(n):
                    if u[j] > umax:
                        umax = u[j]
                if umax > 0:
                    chosen = -1
                    for j in range(n):
                        if u[j] == umax and deg_v[j] > 0:
                            chosen = j
                            break
                    if chosen >= 0:
                        w[chosen] = 1 - w[chosen]
                        total_flips += 1
                else:
                    # stuck: rollback to original and exit
                    decoded[b, :] = orig_w
                    success[b] = False
                    iters[b] = it
                    syndromes[b, :] = s
                    flips[b] = total_flips
                    finished = True
                    break
            else:
                # flip all at once (synchronous)
                cnt = 0
                for j in range(n):
                    if flip_mask[j] == 1:
                        w[j] = 1 - w[j]
                        cnt += 1
                total_flips += cnt

            # stagnation guard: if w equals last_w, force flip of first umax var
            same = True
            for j in range(n):
                if w[j] != last_w[j]:
                    same = False
                    break
            if same:
                umax = 0
                for j in range(n):
                    if u[j] > umax:
                        umax = u[j]
                if umax > 0:
                    chosen = -1
                    for j in range(n):
                        if u[j] == umax and deg_v[j] > 0:
                            chosen = j
                            break
                    if chosen >= 0:
                        w[chosen] = 1 - w[chosen]
                        total_flips += 1

            # update last_w
            for j in range(n):
                last_w[j] = w[j]

            # reached last iteration?
            if it == max_iter:
                # final syndrome
                s = np.zeros(m, dtype=np.uint8)
                nonzero = 0
                for i in range(m):
                    acc = 0
                    for j in range(n):
                        if H[i, j] != 0:
                            acc ^= (w[j] & 1)
                    s[i] = acc
                    if acc != 0:
                        nonzero += 1

                success[b] = (nonzero == 0)
                iters[b] = it
                syndromes[b, :] = s
                if success[b] and (np.sum(w) <= np.sum(orig_w) + weight_tolerance):
                    decoded[b, :] = w
                else:
                    decoded[b, :] = orig_w
                    success[b] = False
                flips[b] = total_flips
                finished = True
                break

        if not finished:
            # safety: if loop exits unexpectedly, roll back
            decoded[b, :] = orig_w
            success[b] = False
            iters[b] = max_iter
            # recompute final syndrome
            s = np.zeros(m, dtype=np.uint8)
            nonzero = 0
            for i in range(m):
                acc = 0
                for j in range(n):
                    if H[i, j] != 0:
                        acc ^= (w[j] & 1)
                s[i] = acc
                if acc != 0:
                    nonzero += 1
            syndromes[b, :] = s
            flips[b] = total_flips

    return decoded, success, iters, syndromes, flips #decoded, ok, its, syn, flips

if __name__ == "__main__1": # 19.77 sec

    H, A = sample_LDPC(codeword_len, databit_num, density=density, pooling_factor=pooling_factor,
                       noise_level=noise_level, save_noise_free=True)
    A_error_free = read_matrix('error_free_codeword')

    # 원래 codeword matrix 중에 correct codeword의 개수 (error 개수는 10^(-4)으로 고정되어있지만, 그게 한 코드워드에 여러군데 발생했을 수 있음)
    # 이 값을 봐야 몇개를 더 디코딩하는데 성공했는지 알 수 있음
    total_codewords = pooling_factor * codeword_len
    correct_codewords = compare_matrix(A_error_free, A)
    # noisy_codewords = total_codewords - correct_codewords
    print("Codeword generated: noise is added, so correct  prior codeword number is as follows")
    print("Correct / Total = {} / {}".format(correct_codewords,total_codewords) )


    # H = H[:parity_num // 2]  # only use half of H
    H = H[:3*parity_num//4] # only use quarter

    start_time = time.time()
    decoded, ok, its, syn, flips = ldpc_bitflip_majority_decode_numba(H, A, max_iter=50)
    print("Total elapsed time: %s seconds" % round(time.time() - start_time, 3))

    print()
    # compare matrix
    if (A_error_free == decoded).all():
        print("Correctly recovered!")
    else:
        correct_guess = 0
        for i in range(total_codewords):
            if (decoded == A_error_free[i]).all(axis=1).any():
                correct_guess += 1
        print("failed decoding: {} / {}".format(total_codewords - ok.sum(),total_codewords)) # ok는 decoding 성공한 word개수를 뜻함. 즉 decoder가 생각하기에 자신이 decoding 실패한 codeword개수가 이 값임
        print("Correct codewords after decoding: ")
        print("Correct / Total = {} / {}".format(correct_guess,total_codewords) ) # 전체 중에 디코딩 성공한 거 개수
        print("Recovered {} more correct codewords".format(correct_guess - correct_codewords))







