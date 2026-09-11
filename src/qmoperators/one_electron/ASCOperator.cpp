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

#include "ASCOperator.h"

#include <MRCPP/Gaussians>
#include <MRCPP/Printer>
#include <MRCPP/Timer>
#include <MRCPP/utils/CompFunction.h>

#include "utils/gto_utils/Intgrl.h"
#include "utils/gto_utils/OrbitalExp.h"
#include "utils/math_utils.h"
#include "utils/print_utils.h"

using mrcpp::GaussExp;
using mrcpp::Printer;
using mrcpp::Timer;

namespace mrchem {

// namespace {

/** @brief Construct a 2C CompFunctionVector from a set of 2C atomic GTO (in other words, simply project the Gaussian spinors into trees)
 *  @param bas_file: basis set file directory
 *  @param coef_file: Coeff matrix for the 2C GTO spinor
 *  @param proj_prec: precision of the tree
 *  @param screen: screening parameter
 *  @param coeff_thrs: 
 *
 *  Build the N_ao complex 2-component (alpha, beta) spinors for one basis/coefficient pair.
 *  Each AO is projected into MW space once and cached; every spinor component is then a cheap
 *  complex-coefficient linear combination of (deep copies of) those cached real trees.
 */
std::shared_ptr<mrcpp::CompFunctionVector> project_spinor_set(const std::string &bas_file, const std::string &coef_file, double proj_prec, double screen, double coeff_thrs) {
    gto_utils::Intgrl intgrl(bas_file);
    gto_utils::OrbitalExp ao_exp(intgrl);
    int nAO = ao_exp.size();

    ComplexMatrix C = math_utils::read_matrix_file_cplx(coef_file);
    if (C.rows() != 2 * nAO || C.cols() != nAO) MSG_ABORT("Coupling coefficient matrix must be (2*N_ao x N_ao): stacked alpha/beta rows, one spinor per column");

    // Project each unique real AO into MW space once.
    std::vector<mrcpp::CompFunction<3>> ao_real(nAO);
    for (int j = 0; j < nAO; j++) {
        GaussExp<3> ao_j = ao_exp.getAO(j);
        ao_j.calcScreening(screen);
        ao_real[j] = mrcpp::CompFunction<3>(0, false, 1);
        mrcpp::build_grid(ao_real[j].real(), ao_j);
        mrcpp::project(proj_prec, ao_real[j].real(), ao_j);
    }

    auto spinors = std::make_shared<mrcpp::CompFunctionVector>(nAO);
    for (int i = 0; i < nAO; i++) {
        mrcpp::CompFunction<3> spinor(0, false, 2); //2 component spinor
        spinor.defcomplex();
        for (int c = 0; c < 2; c++) {
            std::vector<ComplexDouble> coefs;
            std::vector<mrcpp::CompFunction<3>> terms;
            for (int j = 0; j < nAO; j++) {
                ComplexDouble c_ij = C(c * nAO + j, i);
                if (std::abs(c_ij) < coeff_thrs) continue;
                mrcpp::CompFunction<3> term;
                mrcpp::deep_copy(term, ao_real[j]); // independent copy: linear_combination mutates its inputs
                coefs.push_back(c_ij);
                terms.push_back(term);
            }
            if (coefs.empty()) {
                spinor.complex(c); // lazily allocates a zero-valued component
                continue;
            }
            mrcpp::CompFunction<3> psi_c;
            mrcpp::linear_combination(psi_c, coefs, terms, proj_prec);
            //insert the linear combination inside the component c of the spinor
            spinor.setCplx(psi_c.CompC[0], c);
            psi_c.CompC[0] = nullptr; // ownership transferred to spinor, avoid double free
        }
        // (*spinors)[i] = spinor;
        mrcpp:deep_copy((*spinors)[i], spinor);
    }
    return spinors;
}

// } // namespace

ASCOperator::ASCOperator(const std::string &large_bas_file,
                         const std::string &large_coef_file,
                         const std::string &small_bas_file,
                         const std::string &small_coef_file,
                         double proj_prec,
                         double screen,
                         double coeff_thrs) {
    Timer timer;
    //project atomic large component
    this->large = project_spinor_set(large_bas_file, large_coef_file, proj_prec, screen, coeff_thrs);
    //orthogonalise the atomic large component w.r.t. themselves (they are orthogonal w.r.t the small comp as well when we obtain them)
    ComplexMatrix S_L = mrcpp::calc_overlap_matrix(*(this->large));
    ComplexMatrix S_Linv = mrcpp::math_utils::hermitian_matrix_pow(S_L, -1.0);
    mrcpp::rotate(*(this->large), S_Linv, proj_prec);
    //project atomic small component (no need for orthogonalisation)
    this->small = project_spinor_set(small_bas_file, small_coef_file, proj_prec, screen, coeff_thrs);
    mrcpp::print::time(2, "Gaussian coupling operator (large component, N=" + std::to_string(this->large->size()) + ")", timer);
    mrcpp::print::time(2, "Gaussian coupling operator (small component, N=" + std::to_string(this->small->size()) + ")", timer);
}

OrbitalVector ASCOperator::operator()(OrbitalVector &inp, int alpha) {
    // <phi^L_i|ket> matrix
    ComplexMatrix matrix_Lket = mrcpp::calc_overlap_matrix(*(this->large), inp); //N_AO x N matrix
    OrbitalVector out(inp.size());
    for (int i=0; i<inp.size(); i++) {
        if (!mrcpp::mpi::my_func(i)) continue;
        auto row_Lket = matrix_Lket.row(i); //not an std::vector<ComplexDouble>, is some Eigen block instead, need to transmute
        std::vector<ComplexDouble> vec_Lket(row_Lket.begin(), row_Lket.end()); //transmuting the block to the needed type
        mrcpp::linear_combination(out[i], vec_Lket, *(this->small),-1.0, false);
    }
    return out;
}

/** @brief compute expectation matrix of X = \sum_i^{N_AO} |phi^S_i><phi^L_i|, with |phi^{S,L}> being small/large component spinors of atomic calculations
*
* @param bra: orbitals on the bra side
* @param ket: orbitals on the ket side
*
* X being a sum of projectors, the expectation matrix of <bra|X|ket> simplifies to computing the overlap matrices of 
* <bra|phi^S_i> and <phi^L_i|ket> and multiplying the two together.
*/
ComplexMatrix ASCOperator::operator()(OrbitalVector &bra, OrbitalVector &ket) {
    Timer t1;
    //<bra|phi^S_i>
    ComplexMatrix left_matrix = mrcpp::calc_overlap_matrix(bra, *(this->small)); //N x N_AO matrix
    //<phi^L_i|ket>
    ComplexMatrix right_matrix = mrcpp::calc_overlap_matrix(*(this->large), ket); //N_AO x N matrix
    std::stringstream o_name;
    o_name << "<i|" << this->name() << "|j>";
    // mrcpp::print::tree(2, o_name.str(), orbital::get_n_nodes(Oket), orbital::get_size_nodes(Oket), t1.elapsed());
    return left_matrix*right_matrix;
}

ComplexDouble ASCOperator::trace(OrbitalVector &Phi) {
    Timer t1;
    //NOTE: there might be a super smart way of avoiding to compute the full matrices but I don't see it right now
    //<bra|phi^S_i>
    ComplexMatrix left_matrix = mrcpp::calc_overlap_matrix(Phi, *(this->small)); //N x N_AO matrix
    //<phi^L_i|ket>
    ComplexMatrix right_matrix = mrcpp::calc_overlap_matrix(*(this->large), Phi); //N_AO x N matrix

    //Compute the trace
    ComplexDouble out = (left_matrix.array() * right_matrix.transpose().array()).sum();

    // std::stringstream o_name;
    // o_name << "<i|" << this->name() << "|j>";
    // mrcpp::print::tree(2, o_name.str(), orbital::get_n_nodes(Oket), orbital::get_size_nodes(Oket), t1.elapsed());
    return out;
}

} // namespace mrchem
