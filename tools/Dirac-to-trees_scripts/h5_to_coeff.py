"""Convert all positive-energy DIRAC 4C spinors of an atom (C1 symmetry) into the coefficient files read by ASCOperator.

Reads the DIRAC checkpoint H.h5 and writes two text files:
    H_large.coef   coefficients of every atomic spinor on the large-component AO basis
    H_small.coef   coefficients of every atomic spinor on the small-component AO basis (restricted kinetic balance)

Each file is a complex matrix of shape (2*N_ao, N_spinors). Column i is the i-th spinor. Rows
[0, N_ao) are the alpha (spin-up) AO coefficients and rows [N_ao, 2*N_ao) are the beta (spin-down)
AO coefficients. The two files list the spinors in the same order, and their large components span
the whole large AO space of the atom, which is what ASCOperator needs to build X = sum |phiS_i><phiL_i|.

AOs are kept Cartesian (as stored by DIRAC: 6 d, 10 f, ...). The C++ side must not convert them to real
solid harmonics (ASCOperator reads them with OrbitalExp(intgrl, /*spherical=*/false)), because the small
component of a p_1/2 spinor lives entirely in the r^2 exp(-a r^2) function that spherical d functions drop.
"""

import argparse
import os

import h5py
import numpy as np

## Coefficients with a smaller modulus are set to zero (removes ~1e-15 numerical noise from DIRAC)
NOISE_THRESHOLD = 1.0e-12

## Largest tolerated deviation of the spinor Gram matrix from the identity
ORTHONORMALITY_TOL = 1.0e-10

## Relative energy gap (to the mean level spacing) below which two consecutive stored MOs are
## considered part of the same degenerate shell, and must therefore be either both kept or both dropped
DEGENERACY_REL_TOL = 1.0e-6

L_LABEL = "spdfghi"

## Fraction of a matrix's own largest coefficient below which an AO shell is judged to carry no real weight
## for a spinor (used to decide whether a whole shell can be pruned, see prune_unneeded_shells()).
## RELATIVE, not an absolute magnitude like NOISE_THRESHOLD: coefficient scale grows with Z (s-shell
## normalization alone goes like Z^{3/2}), so a fixed absolute cutoff tuned against a light atom is wrong
## by orders of magnitude for a heavy one. Also deliberately looser than a plain "> 0" check: DIRAC's own
## SCF/eigensolver leaves residual numerical mixing well above machine precision (e.g. a nominally pure 1s
## AO on H was observed at ~3e-12 relative on the p shell -- not a real physical admixture, just solver noise)
SHELL_PRUNE_TOL = 1.0e-6


def write_complex_matrix_file(path, C):
    """Write a complex matrix in the plain-text format of math_utils::read_matrix_file_cplx.

    @param path  output file name
    @param C     complex ndarray of shape (nRows, nCols)

    File format: first line "nRows nCols", then nRows*nCols lines "real imag", column-major
    (all rows of column 0, then all rows of column 1, ...).
    """
    nrows, ncols = C.shape
    with open(path, "w") as out:
        out.write(f"{nrows} {ncols}\n")
        for i in range(ncols):
            for j in range(nrows):
                out.write(f"{C[j, i].real:.14e} {C[j, i].imag:.14e}\n")


def read_ao_overlap(f, n_ao):
    """Read the AO overlap matrix stored by DIRAC (packed upper triangle, column by column).

    @param f     open h5py.File of the DIRAC checkpoint
    @param n_ao  total number of AOs (large + small)
    @return      real symmetric ndarray (n_ao, n_ao); its large-small block is zero
    """
    packed = f["result/operators/ao_matrices/OVERLAP TFFT"][()]
    S = np.zeros((n_ao, n_ao))
    for j in range(n_ao):
        for i in range(j + 1):
            S[i, j] = S[j, i] = packed[j * (j + 1) // 2 + i]
    return S


def quaternion_to_spinors(q):
    """Convert the quaternion AO coefficients of one stored DIRAC MO into its two Kramers-partner spinors.

    DIRAC stores each coefficient as a quaternion q0 + q1 i + q2 j + q3 k (q[:, 0..3]) and each stored MO
    stands for a Kramers pair. Writing the quaternion as A + B j with A = q0 + i q1 and B = q2 + i q3, the two
    complex 2-component spinors are (see the QTOC routine of DIRAC, ITIM = +1):

        member 1:  alpha = A          beta = -conj(B)
        member 2:  alpha = B          beta = +conj(A)

    Both members are normalized and mutually orthogonal (checked against the AO overlap in main()).

    @param q  ndarray (n_ao, 4), quaternion coefficients of one MO over all AOs (large then small)
    @return   list of two (alpha, beta) tuples, complex ndarrays of shape (n_ao,)
    """
    A = q[:, 0] + 1j * q[:, 1]
    B = q[:, 2] + 1j * q[:, 3]
    return [(A, -np.conj(B)), (B, np.conj(A))]


def read_shells(f, group):
    """Read the shell structure of one DIRAC AO basis.

    @param f      open h5py.File
    @param group  "input/aobasis/1" (large) or "input/aobasis/2" (small)
    @return       list of (l, exponent) per shell, in DIRAC order (all shells here are uncontracted)
    """
    orbmom = f[group + "/orbmom"][()]      # DIRAC stores l + 1
    expo = f[group + "/exponents"][()]
    n_cont = f[group + "/n_cont"][()]
    n_prim = f[group + "/n_prim"][()]
    assert np.all(n_cont == 1) and np.all(n_prim == 1), "only uncontracted shells are supported"
    return [(int(l) - 1, float(e)) for l, e in zip(orbmom, expo)]


def n_cartesian(l):
    """Number of Cartesian AOs of angular momentum l: (l+1)(l+2)/2."""
    return (l + 1) * (l + 2) // 2


def shell_offsets(shells):
    """First AO index of every shell.

    @param shells  list of (l, exponent)
    @return        list of offsets, and the total number of AOs
    """
    offsets, n = [], 0
    for l, _ in shells:
        offsets.append(n)
        n += n_cartesian(l)
    return offsets, n


def rkb_small_shell_order(large_shells, dirac_small_shells):
    """Permutation from the DIRAC small-component AO order to the order built by gto_utils::generate_rkb_basis().

    generate_rkb_basis() loops over the large shells in order and, for each shell of angular momentum l, appends
    a small shell with l+1 and then (if l > 0) a small shell with l-1, both with the exponent of the large shell.
    DIRAC lists the same shells in its own order, so every shell is located by (l, exponent).

    @param large_shells        list of (l, exponent) of the large basis
    @param dirac_small_shells  list of (l, exponent) of the DIRAC small basis
    @return                    (perm, our_shells, source_l): perm[k] is the DIRAC AO index of our small AO k;
                               our_shells is the list of (l, exponent) in our order; source_l[i] is the angular
                               momentum of the LARGE shell that generated our_shells[i] (needed to prune a small
                               shell together with the large shell it came from, see prune_unneeded_shells())
    """
    offsets, n_dirac = shell_offsets(dirac_small_shells)
    used = [False] * len(dirac_small_shells)
    perm, our_shells, source_l = [], [], []
    for l, e in large_shells:
        wanted = [l + 1] + ([l - 1] if l > 0 else [])
        for lw in wanted:
            idx = next(k for k, (ls, es) in enumerate(dirac_small_shells) if not used[k] and ls == lw and abs(es - e) < 1e-8)
            used[idx] = True
            perm.extend(range(offsets[idx], offsets[idx] + n_cartesian(lw)))
            our_shells.append((lw, e))
            source_l.append(l)
    assert all(used), "DIRAC small basis contains shells that RKB does not generate"
    assert len(perm) == n_dirac
    return np.array(perm), our_shells, source_l


def needed_angular_momenta(C, shells, rel_tol=SHELL_PRUNE_TOL):
    """Angular momenta with at least one non-negligible coefficient, over any column of C.

    Only the alpha AO block (rows [0, n_ao)) is inspected; beta shares the same per-shell layout, so an
    all-zero alpha shell is all-zero in beta too for an atomic spinor (checked in prune_unneeded_shells()).

    The cutoff is RELATIVE to the largest coefficient in C, not an absolute magnitude: coefficient scale
    grows with Z (the s-shell normalization alone goes like Z^{3/2}), so an absolute cutoff tuned against a
    light atom can be wildly wrong -- too loose or too tight -- for a heavy one like Au.

    @param C        complex ndarray (2*n_ao, n_spin)
    @param shells   list of (l, exponent) covering rows [0, n_ao)
    @param rel_tol  fraction of the largest coefficient in C below which a coefficient counts as zero
    @return         set of l values that have a non-negligible coefficient somewhere in C
    """
    offsets, n_ao = shell_offsets(shells)
    row_max = np.max(np.abs(C[:n_ao]), axis=1)
    noise = rel_tol * np.max(row_max)
    return {l for o, (l, _) in zip(offsets, shells) if np.max(row_max[o:o + n_cartesian(l)]) > noise}


def prune_unneeded_shells(C_large, C_small, large_shells, our_small_shells, source_l, rel_tol=SHELL_PRUNE_TOL):
    """Drop large/small AO shells that are exactly zero for every kept spinor, by atomic (l, j) selection rules.

    A free atom's Dirac equation separates by (l, j): the large component of an atomic spinor has one
    definite l, and its small component (restricted kinetic balance) only ever mixes l+1 and l-1 relative
    to that. So once the set of large-l values actually present among the kept spinors is known, every large
    shell of a different l, and every small shell generated from a large shell of a different l, is exactly
    zero (up to numerical noise) and can be dropped with no loss of information -- this is a lossless size
    reduction, not an approximation (unlike a magnitude-based cutoff within a kept l).

    @param C_large, C_small   full coefficient matrices, as built in main()
    @param large_shells       list of (l, exponent) of the large AO basis
    @param our_small_shells   list of (l, exponent) of the small AO basis, in our (RKB) order
    @param source_l           source_l from rkb_small_shell_order(), parallel to our_small_shells
    @param rel_tol            fraction of a matrix's own largest coefficient below which a coefficient
                              counts as zero (see needed_angular_momenta()); applied to C_large and
                              C_small separately, since the small component is intrinsically much smaller
                              than the large one (by ~1/2c) and would false-trip a shared absolute scale
    @return  (C_large_pruned, C_small_pruned, large_shells_kept, small_shells_kept, l_needed)
    """
    l_needed = needed_angular_momenta(C_large, large_shells, rel_tol)

    def ao_mask_and_kept_shells(shells, shell_l_for_keep):
        offsets, n_ao = shell_offsets(shells)
        mask = np.zeros(n_ao, dtype=bool)
        kept_shells = []
        for o, shell, keep_l in zip(offsets, shells, shell_l_for_keep):
            l, _ = shell
            if keep_l in l_needed:
                mask[o:o + n_cartesian(l)] = True
                kept_shells.append(shell)
        return mask, kept_shells

    large_mask, large_shells_kept = ao_mask_and_kept_shells(large_shells, [l for l, _ in large_shells])
    small_mask, small_shells_kept = ao_mask_and_kept_shells(our_small_shells, source_l)

    # sanity check on the "exactly zero" claim: nothing dropped should have carried any real weight, in
    # either the alpha or the beta half (beta uses the same per-shell AO layout as alpha, see docstring),
    # relative to that matrix's own scale (large and small live at very different absolute magnitudes)
    for C, mask, tag in ((C_large, large_mask, "large"), (C_small, small_mask, "small")):
        n_ao = len(mask)
        scale = np.max(np.abs(C[:n_ao]))
        dropped = np.concatenate([~mask, ~mask])
        worst = np.max(np.abs(C[dropped])) if dropped.any() else 0.0
        assert worst < rel_tol * scale, (
            f"{tag} shell pruning would drop a non-negligible coefficient "
            f"({worst:.2e}, {worst / scale:.2e} relative to the largest {tag} coefficient)")

    C_large_pruned = C_large[np.concatenate([large_mask, large_mask])]
    C_small_pruned = C_small[np.concatenate([small_mask, small_mask])]
    return C_large_pruned, C_small_pruned, large_shells_kept, small_shells_kept, l_needed


def check_ground_state_kinetic_balance(alpha_small, beta_small, our_shells, atol_rel=1.0e-6):
    """Check the sign/conjugation of beta for a real spin-up s large component (the lowest spinor).

    For psi_S = -i/(2c) (sigma.p) psi_L with psi_L real and in the alpha channel,
        S_alpha ~ z g(r)          (p_z AOs only)
        S_beta  ~ (x + i y) g(r)  (p_x and p_y AOs)
    so in every small p shell (AO order x, y, z):
        alpha_px = alpha_py = beta_pz = 0,  beta_px = alpha_pz,  beta_py = i alpha_pz.

    @param alpha_small, beta_small  complex ndarrays, small-component coefficients in our AO order
    @param our_shells               list of (l, exponent) in our AO order
    @param atol_rel                 tolerance relative to the largest |alpha_pz|
    """
    offsets, _ = shell_offsets(our_shells)
    p_offsets = [o for o, (l, _) in zip(offsets, our_shells) if l == 1]
    scale = max(abs(alpha_small[o + 2]) for o in p_offsets)
    tol = atol_rel * scale
    for o in p_offsets:
        ax, ay, az = alpha_small[o:o + 3]
        bx, by, bz = beta_small[o:o + 3]
        assert abs(ax) < tol and abs(ay) < tol and abs(bz) < tol, f"p shell at AO {o}: unexpected nonzero coefficient"
        assert abs(bx - az) < tol, f"p shell at AO {o}: beta_px != alpha_pz"
        assert abs(by - 1j * az) < tol, f"p shell at AO {o}: beta_py != i alpha_pz"


def check_no_split_shell(eps, n_po, n_pairs):
    """Abort if the requested cutoff falls inside a degenerate shell (e.g. half of a p3/2 quartet).

    Compares the gap at the cutoff to the mean level spacing among the kept positive-energy MOs;
    a gap far smaller than that mean is taken to mean the cutoff splits a degenerate shell.

    @param eps      full eigenvalue array (stored-MO order)
    @param n_po     number of negative-energy stored MOs (offset of the positive-energy branch)
    @param n_pairs  number of positive-energy stored MOs (Kramers pairs) requested
    """
    n_mo = len(eps)
    if n_pairs >= n_mo - n_po:
        return  # taking everything, nothing to split
    kept = eps[n_po:n_po + n_pairs]
    mean_gap = (kept[-1] - kept[0]) / max(len(kept) - 1, 1)
    gap = eps[n_po + n_pairs] - eps[n_po + n_pairs - 1]
    assert gap > DEGENERACY_REL_TOL * mean_gap or mean_gap == 0.0, (
        f"n_pairs={n_pairs} cuts a degenerate shell in half (gap {gap:.3e} at the cutoff vs. mean "
        f"spacing {mean_gap:.3e} among the kept MOs); adjust n_pairs by +/-1 (or more) to land on a shell boundary"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5file", nargs="?", default="H.h5", help="DIRAC checkpoint to read (default: H.h5)")
    parser.add_argument("n_pairs", nargs="?", type=int, default=None,
                         help="number of positive-energy stored MOs (Kramers pairs) to keep, lowest energy "
                              "first, 2 spinors each (default: all). E.g. 4 for Au's 1s,2s,2p1/2,2p3/2, "
                              "since 2p3/2 alone is 2 stored MOs")
    args = parser.parse_args()
    tag = os.path.splitext(os.path.basename(args.h5file))[0]

    with h5py.File(args.h5file, "r") as f:
        mo = f["result/wavefunctions/scf/mobasis"]
        orb = mo["orbitals"][()]
        n_basis = int(mo["n_basis"][0])   # large + small AOs
        n_mo = int(mo["n_mo"][0])         # stored MOs (Kramers pairs), negative-energy ones first
        n_po = int(mo["n_po"][0])         # number of negative-energy (positronic) stored MOs
        nz = int(mo["nz"][0])             # quaternion components (4)
        eps = mo["eigenvalues"][()]
        large_shells = read_shells(f, "input/aobasis/1")
        dirac_small_shells = read_shells(f, "input/aobasis/2")
        S = read_ao_overlap(f, n_basis)

    assert nz == 4
    assert [l for l, _ in large_shells] == sorted(l for l, _ in large_shells), \
        "large shells must be ordered by angular momentum (as written by h5_to_bas.py)"
    _, n_ao_large = shell_offsets(large_shells)
    perm, our_small_shells, source_l = rkb_small_shell_order(large_shells, dirac_small_shells)
    n_ao_small = len(perm)
    assert n_ao_large + n_ao_small == n_basis

    n_pairs = args.n_pairs if args.n_pairs is not None else (n_mo - n_po)
    print("stored MO  energy (au)   [x] = kept")
    for col in range(n_po, n_mo):
        print(f"{col - n_po:9d}  {eps[col]:12.6f}   {'x' if col - n_po < n_pairs else ' '}")
    check_no_split_shell(eps, n_po, n_pairs)

    # q[ao, mo, k]: k-th quaternion component of the coefficient of AO `ao` in stored MO `mo` (Fortran order)
    q = orb.reshape((n_basis, n_mo, nz), order="F")

    # every kept positive-energy stored MO gives two Kramers-partner spinors
    spinors, energies = [], []
    for col in range(n_po, n_po + n_pairs):
        for alpha, beta in quaternion_to_spinors(q[:, col, :]):
            spinors.append((alpha, beta))
            energies.append(eps[col])
    n_spin = len(spinors)

    # validation: the spinors must be orthonormal in the AO overlap metric (alpha and beta share S)
    G = np.array([[np.vdot(a1, S @ a2) + np.vdot(b1, S @ b2) for (a2, b2) in spinors] for (a1, b1) in spinors])
    dev = np.max(np.abs(G - np.eye(n_spin)))
    assert dev < ORTHONORMALITY_TOL, f"spinors are not orthonormal (max deviation {dev:.2e})"

    C_large = np.zeros((2 * n_ao_large, n_spin), dtype=complex)
    C_small = np.zeros((2 * n_ao_small, n_spin), dtype=complex)
    for i, (alpha, beta) in enumerate(spinors):
        # large AOs: DIRAC order == our order (both from aobasis/1)
        C_large[0:n_ao_large, i] = alpha[:n_ao_large]
        C_large[n_ao_large:, i] = beta[:n_ao_large]
        # small AOs: DIRAC order -> RKB order of generate_rkb_basis()
        C_small[0:n_ao_small, i] = alpha[n_ao_large:][perm]
        C_small[n_ao_small:, i] = beta[n_ao_large:][perm]

    C_large[np.abs(C_large) < NOISE_THRESHOLD] = 0.0
    C_small[np.abs(C_small) < NOISE_THRESHOLD] = 0.0

    # the lowest spinor is the 1s_1/2 ground state, for which beta follows from kinetic balance
    check_ground_state_kinetic_balance(C_small[0:n_ao_small, 0], C_small[n_ao_small:, 0], our_small_shells)

    # exact (lossless) AO pruning: drop large/small shells that are exactly zero for every kept spinor,
    # by atomic (l, j) selection rules -- see prune_unneeded_shells()'s docstring
    C_large, C_small, large_shells_kept, small_shells_kept, l_needed = prune_unneeded_shells(
        C_large, C_small, large_shells, our_small_shells, source_l)
    print(f"angular momenta needed by the kept spinors: {sorted(l_needed)} ({[L_LABEL[l] for l in sorted(l_needed)]})")
    print(f"large AOs: {n_ao_large} -> {C_large.shape[0] // 2} "
          f"({len(large_shells)} -> {len(large_shells_kept)} shells)")
    print(f"small AOs: {n_ao_small} -> {C_small.shape[0] // 2} "
          f"({len(our_small_shells)} -> {len(small_shells_kept)} shells)")

    large_path, small_path = f"{tag}_large.coef", f"{tag}_small.coef"
    write_complex_matrix_file(large_path, C_large)
    write_complex_matrix_file(small_path, C_small)
    print(f"wrote {large_path} {C_large.shape}, {small_path} {C_small.shape}")
    print(f"NOTE: {large_path} only has the shells listed above -- the paired .bas file passed to MRChem "
          f"must be filtered to the same large shells (same order), e.g. with h5_to_bas.py's --keep-l "
          f"{' '.join(str(l) for l in sorted(l_needed))}")
    print(f"max deviation of the spinor Gram matrix from identity: {dev:.2e}")
    print("spinor  energy          <L|L>      <S|S>")
    for i, (alpha, beta) in enumerate(spinors):
        nl = np.real(np.vdot(alpha[:n_ao_large], S[:n_ao_large, :n_ao_large] @ alpha[:n_ao_large]) + np.vdot(beta[:n_ao_large], S[:n_ao_large, :n_ao_large] @ beta[:n_ao_large]))
        ns = np.real(np.vdot(alpha[n_ao_large:], S[n_ao_large:, n_ao_large:] @ alpha[n_ao_large:]) + np.vdot(beta[n_ao_large:], S[n_ao_large:, n_ao_large:] @ beta[n_ao_large:]))
        print(f"{i:4d}  {energies[i]:12.6f}  {nl:9.6f}  {ns:.4e}")


if __name__ == "__main__":
    main()
