/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2023 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
 *
 * This file is part of MRChem.
 *
 * MRChem is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MRChem is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MRChem.  If not, see <https://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to MRChem, see:
 * <https://mrchem.readthedocs.io/>
 */

#include "X2COperator.h"

#include <MRCPP/Gaussians>
#include <MRCPP/Printer>
#include <MRCPP/Timer>

#include "qmoperators/QMPotential.h"
#include "utils/gto_utils/Intgrl.h"
#include "utils/gto_utils/OrbitalExp.h"
#include "utils/math_utils.h"
#include "utils/print_utils.h"

using mrcpp::GaussExp;
using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {

X2COperator::X2COperator(const std::string &bas_file, const std::string &mat_file, double proj_prec, double screen, const std::string &name) {
    Timer timer;

    // Contracted GTO AO basis, one GaussExp<3> per AO, as read from the Gaussian-code basis file
    gto_utils::Intgrl intgrl(bas_file);
    gto_utils::OrbitalExp gto_exp(intgrl);

    // Operator matrix in that AO basis (same layout as an SCF density matrix)
    DoubleMatrix C = math_utils::read_matrix_file(mat_file);
    if (C.rows() != gto_exp.size() || C.cols() != gto_exp.size()) MSG_ABORT("Gaussian coupling matrix does not match AO basis size");

    // chi(r) = sum_ij C_ij * chi_i(r) * chi_j(r), expanded analytically as a sum of Gaussians
    GaussExp<3> chi_exp = gto_exp.getDens(C);
    chi_exp.calcScreening(screen);

    // Project the Gaussian expansion onto the MW representation
    auto k = std::make_shared<QMPotential>(1); // int adap = 1 here => adaptative grid. Potentially not required.
    mrcpp::project(*k, chi_exp, proj_prec);
    if (k->hasReal()) k->real().crop(proj_prec);

    RankZeroOperator &chi = (*this);
    chi = k;
    chi.name() = name;
    print_utils::qmfunction(2, "Gaussian coupling operator (" + chi.name() + ")", *k, timer);
}

} // namespace mrchem
