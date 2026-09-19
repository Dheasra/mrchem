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

import h5py
import numpy as np

## Coefficients with a smaller modulus are set to zero (removes ~1e-15 numerical noise from DIRAC)
NOISE_THRESHOLD = 1.0e-12

## Largest tolerated deviation of the spinor Gram matrix from the identity
ORTHONORMALITY_TOL = 1.0e-10


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
    @return                    (perm, our_shells): perm[k] is the DIRAC AO index of our small AO k,
                               our_shells is the list of (l, exponent) in our order
    """
    offsets, n_dirac = shell_offsets(dirac_small_shells)
    used = [False] * len(dirac_small_shells)
    perm, our_shells = [], []
    for l, e in large_shells:
        wanted = [l + 1] + ([l - 1] if l > 0 else [])
        for lw in wanted:
            idx = next(k for k, (ls, es) in enumerate(dirac_small_shells) if not used[k] and ls == lw and abs(es - e) < 1e-8)
            used[idx] = True
            perm.extend(range(offsets[idx], offsets[idx] + n_cartesian(lw)))
            our_shells.append((lw, e))
    assert all(used), "DIRAC small basis contains shells that RKB does not generate"
    assert len(perm) == n_dirac
    return np.array(perm), our_shells


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


def main():
    with h5py.File("H.h5", "r") as f:
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
    perm, our_small_shells = rkb_small_shell_order(large_shells, dirac_small_shells)
    n_ao_small = len(perm)
    assert n_ao_large + n_ao_small == n_basis

    # q[ao, mo, k]: k-th quaternion component of the coefficient of AO `ao` in stored MO `mo` (Fortran order)
    q = orb.reshape((n_basis, n_mo, nz), order="F")

    # every positive-energy stored MO gives two Kramers-partner spinors
    spinors, energies = [], []
    for col in range(n_po, n_mo):
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

    write_complex_matrix_file("H_large.coef", C_large)
    write_complex_matrix_file("H_small.coef", C_small)
    print(f"wrote H_large.coef {C_large.shape}, H_small.coef {C_small.shape}")
    print(f"max deviation of the spinor Gram matrix from identity: {dev:.2e}")
    print("spinor  energy          <L|L>      <S|S>")
    for i, (alpha, beta) in enumerate(spinors):
        nl = np.real(np.vdot(alpha[:n_ao_large], S[:n_ao_large, :n_ao_large] @ alpha[:n_ao_large]) + np.vdot(beta[:n_ao_large], S[:n_ao_large, :n_ao_large] @ beta[:n_ao_large]))
        ns = np.real(np.vdot(alpha[n_ao_large:], S[n_ao_large:, n_ao_large:] @ alpha[n_ao_large:]) + np.vdot(beta[n_ao_large:], S[n_ao_large:, n_ao_large:] @ beta[n_ao_large:]))
        print(f"{i:4d}  {energies[i]:12.6f}  {nl:9.6f}  {ns:.4e}")


if __name__ == "__main__":
    main()
