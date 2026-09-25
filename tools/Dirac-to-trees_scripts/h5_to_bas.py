import h5py
import numpy as np

L_LABEL = "spdfghi"


def write_bas_from_aobasis(h5path, aobasis_index, out_path, symbol, charge):
    with h5py.File(h5path, "r") as f:
        g = f[f"input/aobasis/{aobasis_index}"]
        angular = int(g["angular"][0])
        if angular != 1:
            raise ValueError(f"aobasis {aobasis_index}: expected Cartesian (angular=1), got {angular}")

        n_shells = int(g["n_shells"][0])
        orbmom = g["orbmom"][()]      # 1-indexed: 1=s, 2=p, 3=d, ... (empirically confirmed against n_ao)
        n_prim = g["n_prim"][()]
        n_cont = g["n_cont"][()]
        expo = g["exponents"][()]
        coef = g["contractions"][()]

        # group shell entries by angular momentum, preserving DIRAC's order within each l
        by_l = {}
        expo_off = 0
        coef_off = 0
        for s in range(n_shells):
            l = orbmom[s] - 1
            np_ = int(n_prim[s])
            nc_ = int(n_cont[s])
            e_block = expo[expo_off:expo_off + np_]
            c_block = coef[coef_off:coef_off + np_ * nc_].reshape(np_, nc_)  # assumed primitive-major
            by_l.setdefault(l, []).append((e_block, c_block))
            expo_off += np_
            coef_off += np_ * nc_

        ls_sorted = sorted(by_l.keys())
        funcs_per_shell = [len(by_l[l]) for l in ls_sorted]

        with open(out_path, "w") as out:
            out.write(f"Converted from {h5path} (aobasis {aobasis_index})\n")
            out.write("1\n")
            out.write(f"{charge:.1f} 1 {len(ls_sorted)}\n")
            out.write(" ".join(str(n) for n in funcs_per_shell) + "\n")
            out.write(f"{symbol} 0.0000000000 0.0000000000 0.0000000000\n")
            for l in ls_sorted:
                for e_block, c_block in by_l[l]:
                    np_, nc_ = c_block.shape
                    out.write(f"{np_} {nc_}\n")
                    for p in range(np_):
                        row = " ".join(f"{c_block[p, c]:.10f}" for c in range(nc_))
                        out.write(f"{e_block[p]:.10f} {row}\n")

        print(f"wrote {out_path}: {len(ls_sorted)} l-values {[L_LABEL[l] for l in ls_sorted]}, "
              f"funcs_per_shell={funcs_per_shell}, n_ao(file)={int(g['n_ao'][0])}")


if __name__ == "__main__":
    write_bas_from_aobasis("H.h5", 1, "H_large.bas", symbol="H", charge=1.0)
    write_bas_from_aobasis("H.h5", 2, "H_small_dirac_rkb.bas", symbol="H", charge=1.0)  # reference only, not fed to MRChem
