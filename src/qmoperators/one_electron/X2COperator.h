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

#pragma once

#include "tensor/RankZeroOperator.h"
#include "qmoperators/one_electron/CouplingOperator.h"
#include <string>

namespace mrchem {

class QMPotential;

/**
 * @class X2COperator
 * @brief Implements a Gaussian amfX2C operator, that is a sum of Gaussian functions approximating the coupling
 * operator in Dirac theory.
 * Coupling operator R(r) = sum_ij C_ij * chi_i(r) * chi_j(r), where chi_i are contracted
 * GTOs read from a Gaussian-code basis set file and C is a matrix representation of the operator
 * in that AO basis (e.g. produced by an external Gaussian-basis quantum chemistry code). The
 * expansion is built analytically as a sum of Gaussians (MRCPP GaussExp) and then projected onto
 * the MW representation, avoiding numerical differentiation/integration of the Gaussian basis.
 */
class X2COperator final : public CouplingOperator {
public:
    /**
     * @param bas_file Basis set file (LSDalton/Intgrl format) defining the GTO basis the matrix is expressed in.
     * @param mat_file File holding the C_ij matrix, in the AO ordering produced by that basis file.
     * @param proj_prec Precision of the MW projection.
     * @param screen GTO screening in standard deviations (negative disables screening).
     * @param name Name assigned to the resulting operator.
     */
    X2COperator(const std::string &bas_file, const std::string &mat_file, double proj_prec,  double screen, const std::string &name = "R");
};

} // namespace mrchem