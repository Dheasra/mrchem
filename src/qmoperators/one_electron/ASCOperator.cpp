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

#include "chemistry/Nucleus.h"
#include "utils/gto_utils/AOBasis.h"
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
// std::shared_ptr<mrcpp::CompFunctionVector> project_spinor_set(const std::string &bas_file, const std::string &coef_file, double proj_prec, double screen, double coeff_thrs) {
//     gto_utils::Intgrl intgrl(bas_file);
//     gto_utils::OrbitalExp ao_exp(intgrl);
//     int nAO = ao_exp.size();

//     ComplexMatrix C = math_utils::read_matrix_file_cplx(coef_file);
//     if (C.rows() != 2 * nAO || C.cols() != nAO) MSG_ABORT("Coupling coefficient matrix must be (2*N_ao x N_ao): stacked alpha/beta rows, one spinor per column");

//     // Project each unique real AO into MW space once.
//     std::vector<mrcpp::CompFunction<3>> ao_real(nAO);
//     for (int j = 0; j < nAO; j++) {
//         GaussExp<3> ao_j = ao_exp.getAO(j);
//         ao_j.calcScreening(screen);
//         ao_real[j] = mrcpp::CompFunction<3>(0, false, 1);
//         mrcpp::build_grid(ao_real[j].real(), ao_j);
//         mrcpp::project(proj_prec, ao_real[j].real(), ao_j);
//     }

//     auto spinors = std::make_shared<mrcpp::CompFunctionVector>(nAO);
//     for (int i = 0; i < nAO; i++) {
//         mrcpp::CompFunction<3> spinor(0, false, 2); //2 component spinor
//         spinor.defcomplex();
//         for (int c = 0; c < 2; c++) {
//             std::vector<ComplexDouble> coefs;
//             std::vector<mrcpp::CompFunction<3>> terms;
//             for (int j = 0; j < nAO; j++) {
//                 ComplexDouble c_ij = C(c * nAO + j, i);
//                 if (std::abs(c_ij) < coeff_thrs) continue;
//                 mrcpp::CompFunction<3> term;
//                 mrcpp::deep_copy(term, ao_real[j]); // independent copy: linear_combination mutates its inputs
//                 coefs.push_back(c_ij);
//                 terms.push_back(term);
//             }
//             if (coefs.empty()) {
//                 spinor.complex(c); // lazily allocates a zero-valued component
//                 continue;
//             }
//             mrcpp::CompFunction<3> psi_c;
//             mrcpp::linear_combination(psi_c, coefs, terms, proj_prec);
//             //insert the linear combination inside the component c of the spinor
//             spinor.setCplx(psi_c.CompC[0], c);
//             psi_c.CompC[0] = nullptr; // ownership transferred to spinor, avoid double free
//         }
//         // (*spinors)[i] = spinor;
//         mrcpp:deep_copy((*spinors)[i], spinor);
//     }
//     return spinors;
// }

// Assemble the molecule-wide set of complex 2-component (alpha, beta) spinors from one
// (bas_file, coef_file) pair per atom. Each atom's basis is read at whatever coordinate its file
// contains and then translated to its real position in nucs (mirrors
// initial_guess::gto::project_ao's Intgrl::getNucleus(0).setCoord()). Every unique (translated) AO
// is projected into MW space once and cached; since each atom's coefficient matrix comes from an
// independent isolated-atom calculation, atoms are combined block-diagonally: atom k's spinors are
// linear combinations of atom k's own AOs only.
std::shared_ptr<mrcpp::CompFunctionVector> project_molecular_spinor_set(const Nuclei &nucs,
                                                                         const std::vector<std::string> &bas_files,
                                                                         const std::vector<std::string> &coef_files,
                                                                         double proj_prec,
                                                                         double screen,
                                                                         double coeff_thrs) {
    int nAtoms = nucs.size();
    if (static_cast<int>(bas_files.size()) != nAtoms || static_cast<int>(coef_files.size()) != nAtoms)
        MSG_ABORT("Need exactly one basis file and one coefficient file per atom");

    std::vector<mrcpp::CompFunction<3>> ao_real; // molecule-wide cache of projected AOs, atom-contiguous
    std::vector<int> ao_offset(nAtoms);          // first molecular AO index belonging to atom k
    std::vector<ComplexMatrix> atom_coefs(nAtoms); // atom k's own (2*N_ao_k x N_ao_k) block

    for (int k = 0; k < nAtoms; k++) {
        gto_utils::Intgrl intgrl(bas_files[k]);
        intgrl.getNucleus(0).setCoord(nucs[k].getCoord()); // translate atomic basis to its real molecular position
        gto_utils::OrbitalExp ao_exp(intgrl);
        int nAO_k = ao_exp.size();

        ComplexMatrix C_k = math_utils::read_matrix_file_cplx(coef_files[k]);
        if (C_k.rows() != 2 * nAO_k || C_k.cols() != nAO_k) MSG_ABORT("Coupling coefficient matrix for atom " + std::to_string(k) + " must be (2*N_ao x N_ao)");
        atom_coefs[k] = C_k;

        ao_offset[k] = ao_real.size();
        for (int j = 0; j < nAO_k; j++) {
            GaussExp<3> ao_j = ao_exp.getAO(j);
            ao_j.calcScreening(screen);
            mrcpp::CompFunction<3> ao(0, false, 1);
            mrcpp::build_grid(ao.real(), ao_j);
            mrcpp::project(proj_prec, ao.real(), ao_j);
            ao_real.push_back(ao);
        }
    }
    int nAO_tot = ao_real.size();

    auto spinors = std::make_shared<mrcpp::CompFunctionVector>(nAO_tot);
    for (int k = 0; k < nAtoms; k++) {
        int nAO_k = static_cast<int>(atom_coefs[k].cols());
        for (int i_local = 0; i_local < nAO_k; i_local++) {
            int i = ao_offset[k] + i_local; // molecular spinor index
            mrcpp::CompFunction<3> spinor(0, false, 2);
            spinor.defcomplex();
            for (int c = 0; c < 2; c++) { // loop over components
                std::vector<ComplexDouble> coefs;
                std::vector<mrcpp::CompFunction<3>> terms;
                for (int j_local = 0; j_local < nAO_k; j_local++) {
                    ComplexDouble c_ij = atom_coefs[k](c * nAO_k + j_local, i_local);
                    if (std::abs(c_ij) < coeff_thrs) continue;
                    mrcpp::CompFunction<3> term;
                    mrcpp::deep_copy(term, ao_real[ao_offset[k] + j_local]); // block-diagonal: atom k's own AOs only
                    coefs.push_back(c_ij);
                    terms.push_back(term); // copy (shallow, shares func_ptr): CompFunction's move ctor is declared but undefined in MRCPP
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
            (*spinors)[i] = spinor;
        }
    }
    return spinors;
}

// Project one atom's (already positioned) AO basis into MW space once, cache it into ao_real, and
// append its block-diagonal contribution (spinors built only from this atom's own AOs) to spinors.
void add_atom_spinors(gto_utils::Intgrl &intgrl,
                       const std::string &coef_file,
                       double proj_prec,
                       double screen,
                       double coeff_thrs,
                       std::vector<mrcpp::CompFunction<3>> &ao_real,
                       mrcpp::CompFunctionVector &spinors) {
    gto_utils::OrbitalExp ao_exp(intgrl);
    int nAO = ao_exp.size();

    ComplexMatrix C = math_utils::read_matrix_file_cplx(coef_file);
    if (C.rows() != 2 * nAO || C.cols() != nAO) MSG_ABORT("Coupling coefficient matrix must be (2*N_ao x N_ao)");

    int offset = ao_real.size();
    for (int j = 0; j < nAO; j++) {
        GaussExp<3> ao_j = ao_exp.getAO(j);
        ao_j.calcScreening(screen);
        mrcpp::CompFunction<3> ao(0, false, 1);
        mrcpp::build_grid(ao.real(), ao_j);
        mrcpp::project(proj_prec, ao.real(), ao_j);
        ao_real.push_back(ao);
    }

    for (int i = 0; i < nAO; i++) {
        mrcpp::CompFunction<3> spinor(0, false, 2);
        spinor.defcomplex();
        for (int c = 0; c < 2; c++) {
            std::vector<ComplexDouble> coefs;
            std::vector<mrcpp::CompFunction<3>> terms;
            for (int j = 0; j < nAO; j++) {
                ComplexDouble c_ij = C(c * nAO + j, i);
                if (std::abs(c_ij) < coeff_thrs) continue;
                mrcpp::CompFunction<3> term;
                mrcpp::deep_copy(term, ao_real[offset + j]); // block-diagonal: this atom's own AOs only
                coefs.push_back(c_ij);
                terms.push_back(term); // copy (shallow, shares func_ptr): CompFunction's move ctor is declared but undefined in MRCPP
            }
            if (coefs.empty()) {
                spinor.complex(c); // lazily allocates a zero-valued component
                continue;
            }
            mrcpp::CompFunction<3> psi_c;
            mrcpp::linear_combination(psi_c, coefs, terms, proj_prec);
            spinor.setCplx(psi_c.CompC[0], c);
            psi_c.CompC[0] = nullptr; // ownership transferred to spinor, avoid double free
        }
        spinors.push_back(spinor);
    }
}

std::shared_ptr<mrcpp::CompFunctionVector> project_large_spinor_set(const Nuclei &nucs,
                                                                     const std::vector<std::string> &bas_files,
                                                                     const std::vector<std::string> &coef_files,
                                                                     double proj_prec,
                                                                     double screen,
                                                                     double coeff_thrs) {
    int nAtoms = nucs.size();
    if (static_cast<int>(bas_files.size()) != nAtoms || static_cast<int>(coef_files.size()) != nAtoms)
        MSG_ABORT("Need exactly one large-component basis file and one coefficient file per atom");

    std::vector<mrcpp::CompFunction<3>> ao_real;
    auto spinors = std::make_shared<mrcpp::CompFunctionVector>(0);
    for (int k = 0; k < nAtoms; k++) {
        gto_utils::Intgrl intgrl(bas_files[k]);
        intgrl.getNucleus(0).setCoord(nucs[k].getCoord()); // translate to the real molecular position
        add_atom_spinors(intgrl, coef_files[k], proj_prec, screen, coeff_thrs, ao_real, *spinors);
    }
    return spinors;
}

// Small-component basis is not read from a file: under restricted kinetic balance it is generated
// mechanically from the large-component one (gto_utils::generate_rkb_basis), then translated to
// the same real molecular position as the large component.
std::shared_ptr<mrcpp::CompFunctionVector> project_small_spinor_set(const Nuclei &nucs,
                                                                     const std::vector<std::string> &large_bas_files,
                                                                     const std::vector<std::string> &coef_files,
                                                                     double proj_prec,
                                                                     double screen,
                                                                     double coeff_thrs) {
    int nAtoms = nucs.size();
    if (static_cast<int>(large_bas_files.size()) != nAtoms || static_cast<int>(coef_files.size()) != nAtoms)
        MSG_ABORT("Need exactly one large-component basis file and one small-component coefficient file per atom");

    std::vector<mrcpp::CompFunction<3>> ao_real;
    auto spinors = std::make_shared<mrcpp::CompFunctionVector>(0);
    for (int k = 0; k < nAtoms; k++) {
        gto_utils::Intgrl large_intgrl(large_bas_files[k]);
        large_intgrl.getNucleus(0).setCoord(nucs[k].getCoord()); // translate to the real molecular position
        gto_utils::AOBasis small_basis = gto_utils::generate_rkb_basis(large_intgrl.getAOBasis(0));
        gto_utils::Intgrl small_intgrl({large_intgrl.getNucleus(0)}, {small_basis}); // reuses the already-translated nucleus
        add_atom_spinors(small_intgrl, coef_files[k], proj_prec, screen, coeff_thrs, ao_real, *spinors);
    }
    return spinors;
}


// } // namespace

ASCOperator::ASCOperator(const Nuclei &nucs,
                        const std::vector<std::string> &large_bas_files,
                        const std::vector<std::string> &large_coef_files,
                        const std::vector<std::string> &small_coef_files,
                        double proj_prec,
                        double screen,
                        double coeff_thrs) {
    Timer timer;
    this->large = project_large_spinor_set(nucs, large_bas_files, large_coef_files, proj_prec, screen, coeff_thrs);
    this->small = project_small_spinor_set(nucs, large_bas_files, small_coef_files, proj_prec, screen, coeff_thrs);
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
