# import h5py
# import numpy as np

# OCC_COL = 9  # occupied H 1s spinor, 0-indexed into the 18 mobasis columns


# def write_complex_matrix_file(path, C):
#     """C: complex ndarray (nRows, nCols). Matches math_utils::read_complex_matrix_file:
#     'nRows nCols' header, then one 'real imag' per line, column-major fill."""
#     nrows, ncols = C.shape
#     with open(path, "w") as out:
#         out.write(f"{nrows} {ncols}\n")
#         for i in range(ncols):
#             for j in range(nrows):
#                 out.write(f"{C[j, i].real:.14e} {C[j, i].imag:.14e}\n")


# def main():
#     with h5py.File("H.h5", "r") as f:
#         orb = f["result/wavefunctions/scf/mobasis/orbitals"][()]
#         n_basis = int(f["result/wavefunctions/scf/mobasis/n_basis"][0])
#         n_mo = int(f["result/wavefunctions/scf/mobasis/n_mo"][0])
#         nz = int(f["result/wavefunctions/scf/mobasis/nz"][0])
#         n_ao_large = int(f["input/aobasis/1/n_ao"][0])
#         n_ao_small = int(f["input/aobasis/2/n_ao"][0])

#     assert nz == 4
#     q = orb.reshape((n_basis, n_mo, nz), order="F")

#     # validated extraction: alpha channel = q0 + i*q1, beta channel = 0
#     # (see conversation: confirmed against the AO overlap matrix, norm = 0.999991;
#     #  q2/q3 are exactly zero for the large-component rows of this column, consistent
#     #  with a single-Kramers-partner / single-spin-channel kappa=-1 (s_1/2) state)
#     c_all = q[:, OCC_COL, 0] + 1j * q[:, OCC_COL, 1]

#     c_large = c_all[:n_ao_large]                        # DIRAC order == our order (both from aobasis/1 directly)
#     c_small_dirac = c_all[n_ao_large:n_ao_large + n_ao_small]  # DIRAC order: [s(1), p(18), d(6 Cartesian)]

#     # DIRAC stores the d-shell as 6 Cartesian components (schema: angular=1), but our
#     # OrbitalExp always spherical-transforms Cartesian shells on read (Cartesian d -> 5
#     # spherical), so our small-component AO count is 24, not 25. Verified the 6 raw
#     # Cartesian d-coefficients for this spinor are all ~1e-15 (machine zero -- this state's
#     # large component has no p-character, so its RKB-generated d-shell partner carries no
#     # weight), so any 6->5 reduction is safe here without implementing the actual
#     # Cartesian->spherical transform: just drop one of the six zero entries.
#     d_block = c_small_dirac[19:25]
#     assert np.max(np.abs(d_block)) < 1e-10, "d-shell not negligible, need real Cartesian->spherical transform"

#     # permute DIRAC's [s, p x6, d(5 after drop)] into our generate_rkb_basis order [p x6, s, d]
#     n_ao_small_sph = n_ao_small - 1  # 24: Cartesian d (6) -> spherical d (5)
#     c_small = np.empty(n_ao_small_sph, dtype=complex)
#     c_small[0:18] = c_small_dirac[1:19]   # the 6 p-blocks (from the 6 large s-shells)
#     c_small[18] = c_small_dirac[0]        # the s (from the large p-shell)
#     c_small[19:24] = d_block[0:5]         # the d (from the large p-shell), 5 of the 6 zero entries

#     # build the (2*N_ao x N_ao) files our GaussCouplingOperator expects.
#     # Only ONE column (0) holds real data (the occupied spinor); the other 8 are
#     # placeholder zeros -- the p-character virtuals were NOT reliably extractable
#     # (see conversation), so they are intentionally left blank rather than wrong.
#     C_large = np.zeros((2 * n_ao_large, n_ao_large), dtype=complex)
#     C_large[0:n_ao_large, 0] = c_large          # alpha
#     # beta rows [n_ao_large:2*n_ao_large, 0] stay zero

#     C_small = np.zeros((2 * n_ao_small_sph, n_ao_small_sph), dtype=complex)
#     C_small[0:n_ao_small_sph, 0] = c_small      # alpha
#     # beta rows 
#     C_small_beta = q[:, OCC_COL, 2] + 1j * q[:, OCC_COL, 3]
#     C_small[n_ao_small_sph:, 0] = -np.conj(C_small_beta)

#     write_complex_matrix_file("H_large.coef", C_large)
#     write_complex_matrix_file("H_small.coef", C_small)
#     print(f"wrote H_large.coef {C_large.shape}, H_small.coef {C_small.shape}")
#     print("column 0 is the only physically meaningful spinor; columns 1-8 are zero placeholders")


# if __name__ == "__main__":
#     main()

"""Convert one occupied DIRAC 4C spinor (H atom, C1 symmetry) into the coefficient files read by ASCOperator.

Reads the DIRAC checkpoint H.h5 and writes two text files:
    H_large.coef   coefficients of the spinor on the large-component AO basis
    H_small.coef   coefficients of the spinor on the small-component AO basis (restricted kinetic balance)

Each file is a complex matrix of shape (2*N_ao, N_ao), see write_complex_matrix_file(). Column i is
the i-th spinor. Rows [0, N_ao) are the alpha (spin-up) AO coefficients and rows [N_ao, 2*N_ao) are
the beta (spin-down) AO coefficients. Only column 0 is filled (the occupied spinor), the other columns are zero.

Only the occupied spinor is converted. The quaternion -> complex reconstruction below is validated
for that column only (see quaternion_to_spinor()).
"""

import h5py
import numpy as np

## Index (0-based, among the n_mo columns of the DIRAC mobasis) of the occupied 1s_1/2 spinor of H
OCC_COL = 9

## Coefficients with a smaller modulus are set to zero (removes ~1e-15 numerical noise from DIRAC)
NOISE_THRESHOLD = 1.0e-12


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


def quaternion_to_spinor(q):
    """Convert the quaternion AO coefficients of one DIRAC spinor to complex alpha/beta coefficients.

    DIRAC stores each coefficient as a quaternion q0 + q1 i + q2 j + q3 k (q[:, 0..3]). Writing it as
    A + B j with A = q0 + i q1 and B = q2 + i q3, the spin-up member of the Kramers pair (the spinor
    stored in column 2j-1 of DIRAC's QTOC routine) has

        alpha_mu = A_mu                 (spin-up AO coefficient)
        beta_mu  = -conj(B_mu)          (spin-down AO coefficient, ITIM = +1)

    The sign and the complex conjugation of beta were validated against restricted kinetic balance
    (see check_kinetic_balance()), and the large-component beta comes out ~0 as expected for a
    real 1s spin-up large component.

    @param q  ndarray (n_ao, 4), quaternion coefficients of one MO over all AOs (large then small)
    @return   (alpha, beta), two complex ndarrays of shape (n_ao,)
    """
    A = q[:, 0] + 1j * q[:, 1]
    B = q[:, 2] + 1j * q[:, 3]
    return A, -np.conj(B)


def to_our_small_ordering(c_small_dirac, n_ao_small_sph):
    """Reorder the small-component coefficients of one spin channel from DIRAC's AO order to ours.

    Restricted kinetic balance maps the large basis (6 s-shells + 1 p-shell) onto small AOs:
        DIRAC order : [ s (1), p (6 shells x 3: x,y,z), d (6 Cartesian) ]   -> 25 functions
        our order   : [ p (6 shells x 3: x,y,z), s (1), d (5 spherical) ]   -> 24 functions
    (our order comes from gto_utils::generate_rkb_basis(): for each large shell in turn, first the
    l+1 small shell, then the l-1 one).

    OrbitalExp always converts Cartesian d to 5 spherical d functions on read. The 6 Cartesian d
    coefficients of this spinor are all zero (the small d shell is generated by the large p shell,
    which is empty for this s-type spinor), so dropping one of them is exact here.

    @param c_small_dirac   complex ndarray (25,), one spin channel in DIRAC order
    @param n_ao_small_sph  number of small AOs on our side (24)
    @return                complex ndarray (24,) in our order
    """
    d_block = c_small_dirac[19:25]
    assert np.max(np.abs(d_block)) < NOISE_THRESHOLD, "d-shell not negligible, need a real Cartesian->spherical transform"

    c_small = np.empty(n_ao_small_sph, dtype=complex)
    c_small[0:18] = c_small_dirac[1:19]  # 6 p-shells generated by the 6 large s-shells
    c_small[18] = c_small_dirac[0]       # s-shell generated by the large p-shell
    c_small[19:24] = d_block[0:5]        # d-shell generated by the large p-shell (all zero here)
    return c_small


def check_kinetic_balance(alpha_small, beta_small, atol_rel=1.0e-6):
    """Check that the small component satisfies kinetic balance for a real spin-up s large component.

    For a real large component f(r) in the alpha channel, psi_S = -i/(2c) (sigma.p) psi_L gives
        S_alpha ~ z g(r)          (p_z AO only)
        S_beta  ~ (x + i y) g(r)  (p_x and p_y AOs)
    so in every p-shell the coefficients must obey
        alpha_px = alpha_py = beta_pz = 0,  beta_px = alpha_pz,  beta_py = i alpha_pz.
    This fixes the sign and conjugation of beta relative to alpha, and the (x, y, z) order of the
    p AOs inside a shell.

    @param alpha_small  complex ndarray (24,), spin-up small coefficients in our order
    @param beta_small   complex ndarray (24,), spin-down small coefficients in our order
    @param atol_rel     tolerance relative to the largest |alpha_pz| of the 6 p-shells
    """
    scale = max(abs(alpha_small[3 * j + 2]) for j in range(6))
    tol = atol_rel * scale
    for j in range(6):
        ax, ay, az = alpha_small[3 * j:3 * j + 3]
        bx, by, bz = beta_small[3 * j:3 * j + 3]
        assert abs(ax) < tol and abs(ay) < tol and abs(bz) < tol, f"p-shell {j}: unexpected nonzero coefficient"
        assert abs(bx - az) < tol, f"p-shell {j}: beta_px != alpha_pz"
        assert abs(by - 1j * az) < tol, f"p-shell {j}: beta_py != i alpha_pz"


def main():
    with h5py.File("H.h5", "r") as f:
        orb = f["result/wavefunctions/scf/mobasis/orbitals"][()]
        n_basis = int(f["result/wavefunctions/scf/mobasis/n_basis"][0])  # large + small AOs (34)
        n_mo = int(f["result/wavefunctions/scf/mobasis/n_mo"][0])        # number of MOs (18)
        nz = int(f["result/wavefunctions/scf/mobasis/nz"][0])            # quaternion components (4)
        n_ao_large = int(f["input/aobasis/1/n_ao"][0])                   # 9
        n_ao_small = int(f["input/aobasis/2/n_ao"][0])                   # 25 (Cartesian d)

    assert nz == 4

    # q[ao, mo, k]: k-th quaternion component of the coefficient of AO `ao` in MO `mo` (Fortran order)
    q = orb.reshape((n_basis, n_mo, nz), order="F")
    alpha, beta = quaternion_to_spinor(q[:, OCC_COL, :])

    # zero out numerical noise, keep the physical (~1e-4 and larger) coefficients
    alpha[np.abs(alpha) < NOISE_THRESHOLD] = 0.0
    beta[np.abs(beta) < NOISE_THRESHOLD] = 0.0

    # large component: DIRAC order == our order (both come from aobasis/1)
    alpha_large = alpha[:n_ao_large]
    beta_large = beta[:n_ao_large]

    # small component: 25 Cartesian AOs in DIRAC order -> 24 spherical AOs in our order
    n_ao_small_sph = n_ao_small - 1
    alpha_small = to_our_small_ordering(alpha[n_ao_large:], n_ao_small_sph)
    beta_small = to_our_small_ordering(beta[n_ao_large:], n_ao_small_sph)
    check_kinetic_balance(alpha_small, beta_small)

    # (2*N_ao x N_ao) coefficient matrices: column 0 = occupied spinor, other columns stay zero
    C_large = np.zeros((2 * n_ao_large, n_ao_large), dtype=complex)
    C_large[0:n_ao_large, 0] = alpha_large
    C_large[n_ao_large:2 * n_ao_large, 0] = beta_large

    C_small = np.zeros((2 * n_ao_small_sph, n_ao_small_sph), dtype=complex)
    C_small[0:n_ao_small_sph, 0] = alpha_small
    C_small[n_ao_small_sph:2 * n_ao_small_sph, 0] = beta_small

    write_complex_matrix_file("H_large.coef", C_large)
    write_complex_matrix_file("H_small.coef", C_small)
    print(f"wrote H_large.coef {C_large.shape}, H_small.coef {C_small.shape}")
    print(f"|alpha_small|^2 = {np.sum(np.abs(alpha_small)**2):.4e}, |beta_small|^2 = {np.sum(np.abs(beta_small)**2):.4e}")
    print("column 0 is the only physically meaningful spinor; columns 1..N-1 are zero placeholders")


if __name__ == "__main__":
    main()
