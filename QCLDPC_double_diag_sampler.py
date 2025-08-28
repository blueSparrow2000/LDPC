import numpy as np
from data_saver import *

'''
Coded with ChatGPT


'''

# ---------- validation ----------
def validate_params(mb, nb, Z, col_degrees, double_diagonal):
    if mb < 1 or nb < 1:
        raise ValueError("mb and nb must be >= 1")
    if Z < 2:
        raise ValueError("Z must be >= 2")
    if len(col_degrees) != nb:
        raise ValueError("col_degrees must have length nb")
    # No-parallel-edge PEG constraint for columns we will PEG-fill
    if double_diagonal:
        # We'll only PEG-fill the first nb-mb columns; check those degrees
        for j in range(nb - mb):
            d = col_degrees[j]
            if d is None:
                raise ValueError("Provide target degree for each info column")
            if d > mb:
                raise ValueError("Each PEG-filled column must satisfy degree <= mb")
    else:
        for d in col_degrees:
            if d is None:
                raise ValueError("Provide degree for all columns when double_diagonal=False")
            if d > mb:
                raise ValueError("Each column degree must satisfy degree <= mb")

# ---------- strict double-diagonal base (no wrap) ----------
def build_base_double_diagonal_linear(mb, nb):
    """
    Last mb columns form a strict lower-bidiagonal (no wrap):
      B[r, nb-mb + r] = 1               (main diagonal)
      B[r, nb-mb + r - 1] = 1 for r>=1  (sub-diagonal)
    => Last parity column has degree 1; others have degree 2.
    """
    B = np.zeros((mb, nb), dtype=np.uint8)
    off = nb - mb
    for r in range(mb):
        B[r, off + r] = 1
        if r >= 1:
            B[r, off + r - 1] = 1
    return B

# ---------- PEG filler (respects existing ones) ----------
def peg_fill_columns(B, fill_cols, target_col_degrees, rng=None):
    rng = np.random.default_rng(rng)
    mb, nb = B.shape
    row_deg = B.sum(axis=1).astype(int)

    var_neighbors = [set(np.flatnonzero(B[:, v])) for v in range(nb)]
    chk_neighbors = [set(np.flatnonzero(B[r, :])) for r in range(mb)]

    from collections import deque
    def bfs_unreached_checks(start_var):
        reached_chk = set()
        q = deque()
        for c in var_neighbors[start_var]:
            reached_chk.add(c); q.append(('chk', c))
        reached_var = {start_var}
        while q:
            kind, node = q.popleft()
            if kind == 'chk':
                for v in chk_neighbors[node]:
                    if v not in reached_var:
                        reached_var.add(v); q.append(('var', v))
            else:
                for c in var_neighbors[node]:
                    if c not in reached_chk:
                        reached_chk.add(c); q.append(('chk', c))
        return reached_chk

    for v in fill_cols:
        dv_target = int(target_col_degrees[v])
        dv_now = int(B[:, v].sum())
        need = dv_target - dv_now
        if need < 0:
            raise ValueError(f"Column {v}: already has degree {dv_now} > target {dv_target}")
        for _ in range(need):
            candidates = [r for r in range(mb) if B[r, v] == 0]
            if not candidates:
                raise ValueError(f"PEG failure: no candidate rows left for column {v}. "
                                 f"Reduce its degree or increase mb.")
            if dv_now == 0:
                min_deg = np.min(row_deg[candidates])
                pool = [r for r in candidates if row_deg[r] == min_deg]
                rsel = rng.choice(pool)
            else:
                reached = bfs_unreached_checks(v)
                pool0 = [r for r in candidates if r not in reached]
                pool = pool0 if pool0 else candidates
                min_deg = np.min(row_deg[pool])
                pool = [r for r in pool if row_deg[r] == min_deg]
                rsel = rng.choice(pool)
            B[rsel, v] = 1
            row_deg[rsel] += 1
            var_neighbors[v].add(rsel)
            chk_neighbors[rsel].add(v)
            dv_now += 1
    return B

# ---------- shift assignment with presets ----------
def assign_circulant_shifts_with_presets(B, Z, S_preset=None, rng=None, max_tries=4000):
    rng = np.random.default_rng(rng)
    mb, nb = B.shape
    S = -np.ones_like(B, dtype=int)
    if S_preset is not None:
        if S_preset.shape != B.shape:
            raise ValueError("S_preset shape mismatch")
        if np.any((S_preset >= 0) & (B == 0)):
            raise ValueError("S_preset has shifts where B==0")
        S[:] = S_preset

    ones_in_row = [np.flatnonzero(B[r, :]) for r in range(mb)]
    ones_in_col = [np.flatnonzero(B[:, c]) for c in range(nb)]
    preset_edges = [(r, c) for r in range(mb) for c in range(nb) if (B[r, c] == 1 and S[r, c] >= 0)]
    free_edges   = [(r, c) for r in range(mb) for c in range(nb) if (B[r, c] == 1 and S[r, c] < 0)]
    rng.shuffle(free_edges)
    edges = preset_edges + free_edges

    for attempt in range(max_tries):
        if attempt:
            for (r, c) in free_edges:
                S[r, c] = -1
            rng.shuffle(free_edges)
            edges = preset_edges + free_edges
        ok = True
        for (r, c) in edges:
            if S[r, c] >= 0:
                s = S[r, c]
                valid = True
                for c2 in ones_in_row[r]:
                    if c2 == c or S[r, c2] < 0: continue
                    for r2 in ones_in_col[c]:
                        if r2 == r or S[r2, c] < 0 or B[r2, c2] == 0 or S[r2, c2] < 0: continue
                        if (s - S[r, c2] - S[r2, c] + S[r2, c2]) % Z == 0:
                            valid = False; break
                    if not valid: break
                if not valid: ok = False; break
                continue
            placed = False
            for s in rng.permutation(Z):
                valid = True
                for c2 in ones_in_row[r]:
                    if c2 == c or S[r, c2] < 0: continue
                    for r2 in ones_in_col[c]:
                        if r2 == r or S[r2, c] < 0 or B[r2, c2] == 0 or S[r2, c2] < 0: continue
                        if (s - S[r, c2] - S[r2, c] + S[r2, c2]) % Z == 0:
                            valid = False; break
                    if not valid: break
                if valid:
                    S[r, c] = s; placed = True; break
            if not placed: ok = False; break
        if ok: return S
    raise RuntimeError("Shift assignment failed; try larger Z or adjust base.")

# ---------- dense H ----------
def circulant_perm(Z, shift):
    I = np.eye(Z, dtype=np.uint8)
    return np.roll(I, shift % Z, axis=1)

def build_qc_ldpc_dense(B, S, Z):
    mb, nb = B.shape
    H = np.zeros((mb*Z, nb*Z), dtype=np.uint8)
    for r in range(mb):
        for c in range(nb):
            if B[r, c] == 1:
                H[r*Z:(r+1)*Z, c*Z:(c+1)*Z] = circulant_perm(Z, int(S[r, c]))
    return H

# ---------- high-level ----------
def generate_qc_ldpc_dense(
    mb, nb, Z, col_degrees, rng=None,
    double_diagonal=True, main_diag_shift=0, sub_diag_shift=0
):
    """
    When double_diagonal=True:
      - Last mb columns are a strict lower-bidiagonal (no wrap).
      - main_diag_shift applies to the main diagonal blocks (default 0 -> identity).
      - sub_diag_shift applies to the sub-diagonal blocks for rows r>=1 (default 0).
      - Only the first nb-mb columns are PEG-filled to the requested degrees.
    """
    validate_params(mb, nb, Z, col_degrees, double_diagonal)
    rng = np.random.default_rng(rng)

    if double_diagonal:
        B = build_base_double_diagonal_linear(mb, nb)
        info_cols = list(range(nb - mb))
        if info_cols:
            B = peg_fill_columns(B, info_cols, col_degrees, rng=rng)

        # preset shifts for crisp diagonals
        S_preset = -np.ones_like(B, dtype=int)
        off = nb - mb
        for r in range(mb):
            S_preset[r, off + r] = int(main_diag_shift) % Z     # main diagonal
            if r >= 1:
                S_preset[r, off + r - 1] = int(sub_diag_shift) % Z  # sub-diagonal
        S = assign_circulant_shifts_with_presets(B, Z, S_preset=S_preset, rng=rng)
    else:
        B = np.zeros((mb, nb), dtype=np.uint8)
        all_cols = list(range(nb))
        B = peg_fill_columns(B, all_cols, col_degrees, rng=rng)
        S = assign_circulant_shifts_with_presets(B, Z, S_preset=None, rng=rng)

    H = build_qc_ldpc_dense(B, S, Z)
    return H, B, S


# Example usage (802.16e-style)
# -----------------------------
if __name__ == "__main__":
    # Base size (choose per target rate R ≈ 1 - mb/nb)
    mb, nb = 4, 12          # block dimension mb x nb / base: 12x24 => rate ≈ 0.5
    Z = 96                   # block size / lifting factor (802.16e uses multiples—pick any; primes also fine)
    # Degrees: first nb-mb are info columns (here, 12 columns). Let’s target degree 3 or 4 typically.
    l_info = 2 # column degree (related to density) : how many 1's in a column
    assert l_info <= mb, "column degree 'l' should not be greater than block row number 'mb' "

    col_degrees = [l_info]*(nb - mb) + [0]*mb  # last mb entries ignored when double_diagonal=True
    H, B, S = generate_qc_ldpc_dense(
        mb, nb, Z, col_degrees, rng=1,
        double_diagonal=True, main_diag_shift=0, sub_diag_shift=0
    )
    print("B (base):\n", B)
    print("\nS (shifts, -1 where B==0):\n", S)
    print("\nH shape:", H.shape, " total ones:", int(H.sum()))

    n = nb*Z
    k = n - mb*Z
    save_image_data(H, filename="n_{}_k_{}.png".format(n,k))


'''
Encoding using QC LDPC

'''