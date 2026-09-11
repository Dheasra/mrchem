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
 * @class ASCOperator (Atomic small component)
 * @brief Implements a Gaussian represented coupling operator, represented the small component as a
 * fixed Gaussian representation for atomic calculations
 * Coupling operator R(r) = sum_ij C_ij * chi_i(r) * chi_j(r), where chi_i are contracted
 * GTOs read from a Gaussian-code basis set file and C is a matrix representation of the operator
 * in that AO basis (e.g. produced by an external Gaussian-basis quantum chemistry code). The
 * expansion is built analytically as a sum of Gaussians (MRCPP GaussExp) and then projected onto
 * the MW representation, avoiding numerical differentiation/integration of the Gaussian basis.
 */
class ASCOperator final : public CouplingOperator {
public:
    /** @brief Construct a 2C CompFunctionVector from a set of 2C atomic GTO (in other words, simply project the Gaussian spinors into trees)
     * @param large_bas_file Basis set file (LSDalton/Intgrl format) for the large component.
     * @param large_coef_file Coefficient file, (nAO_large x 2) matrix: alpha, beta columns.
     * @param small_bas_file Basis set file (LSDalton/Intgrl format) for the small component.
     * @param small_coef_file Coefficient file, (nAO_small x 2) matrix: alpha, beta columns.
     * @param proj_prec Precision of the MW projection.
     * @param screen GTO screening in standard deviations (negative disables screening).
     * @param coeff_thrs Coefficients with magnitude below this are dropped from the linear combination.
     */
    ASCOperator(const std::string &large_bas_file,
                const std::string &large_coef_file,
                const std::string &small_bas_file,
                const std::string &small_coef_file,
                double proj_prec,
                double screen = -1.0,
                double coeff_thrs = mrcpp::MachineZero);

    //Getters
    std::shared_ptr<mrcpp::CompFunctionVector> &getLargeComponents() { return this->large; }
    std::shared_ptr<mrcpp::CompFunctionVector> &getSmallComponents() { return this->small; }

    //operators override
    OrbitalVector operator()(OrbitalVector &inp, int alpha = 0); 
    ComplexMatrix operator()(OrbitalVector &bra, OrbitalVector &ket);
    ComplexDouble trace(OrbitalVector &Phi);
private:
    std::shared_ptr<mrcpp::CompFunctionVector> large{nullptr}; ///< N_ao spinors, comp[0]=alpha, comp[1]=beta
    std::shared_ptr<mrcpp::CompFunctionVector> small{nullptr}; ///< N_ao spinors, comp[0]=alpha, comp[1]=beta
};

} // namespace mrchem