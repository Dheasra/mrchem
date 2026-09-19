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

// Assemble the molecule-wide set of complex 2-component (alpha, beta) spinors from one
// (bas_file, coef_file) pair per atom. Each atom's basis is read at whatever coordinate its file
// contains and then translated to its real position in nucs (mirrors
// initial_guess::gto::project_ao's Intgrl::getNucleus(0).setCoord()). Every unique (translated) AO
// is projected into MW space once and cached; since each atom's coefficient matrix comes from an
// independent isolated-atom calculation, atoms are combined block-diagonally: atom k's spinors are
// linear combinations of atom k's own AOs only.
std::shared_ptr<mrcpp::CompFunctionVector> project_molecular_spinor_set(const Nuclei &nucs, const std::vector<std::string> &bas_files, const std::vector<std::string> &coef_files, double proj_prec, double screen, double coeff_thrs) {
    MSG_WARN("DEPRECATED FUNCTION. Use at your own peril.")
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
void add_atom_spinors(gto_utils::Intgrl &intgrl, const std::string &coef_file, double proj_prec, double screen, double coeff_thrs, std::vector<mrcpp::CompFunction<3>> &ao_real, mrcpp::CompFunctionVector &spinors) {
    // Cartesian AOs (6 d, 10 f, ...): the coefficient files use the DIRAC AO layout, and the restricted
    // kinetic balance small basis needs the r^2 exp(-a r^2) function that real solid harmonics drop.
    gto_utils::OrbitalExp ao_exp(intgrl, false);
    int nAO = ao_exp.size();

    // (2*nAO x nSpinors): rows [0,nAO) alpha AO coefficients, rows [nAO,2*nAO) beta; one column per atomic spinor
    ComplexMatrix C = math_utils::read_matrix_file_cplx(coef_file);
    MSG_INFO("C(0,0) for " << coef_file << " = " << C(0,0)); 
    if (C.rows() != 2 * nAO) MSG_ABORT("Coupling coefficient matrix must have 2*" << nAO << " rows, current format= (" << C.rows() << " x " << C.cols() << ")");
    int nSpinors = C.cols();

    int offset = ao_real.size();
    for (int j = 0; j < nAO; j++) {
        GaussExp<3> ao_j = ao_exp.getAO(j);
        ao_j.calcScreening(screen);
        mrcpp::CompFunction<3> ao(0, false, 1);
        mrcpp::build_grid(ao.real(), ao_j);
        mrcpp::project(proj_prec, ao.real(), ao_j);
        ao_real.push_back(ao);
    }
    MSG_INFO("ao_real[0] norm = " << ao_real[offset].real().getSquareNorm());

    for (int i = 0; i < nSpinors; i++) {
        mrcpp::CompFunction<3> spinor(0, false, 2);
        spinor.defcomplex();
        bool empty_spinor = true; // Keeps track of a spinor being empty or not, to avoid pushing placeholder(zero)-valued spinors to the expansion. Should prevent a size mismatch between large and small components 
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
            empty_spinor = false; //spinor has coefficients
            mrcpp::CompFunction<3> psi_c;
            MSG_INFO("i=" << i << " c=" << c << " coefs.size()=" << coefs.size() << " terms.size()=" << terms.size());
            mrcpp::linear_combination(psi_c, coefs, terms, proj_prec);
            MSG_INFO("psi_c norm after linear_combination = " << psi_c.norm());  // or getSquareNorm() on whichever component is populated
            if (psi_c.isreal()) {
                psi_c.CompC[0]= psi_c.CompD[0]->CopyTreeToComplex();
                delete psi_c.CompD[0];
                psi_c.CompD[0] = nullptr;
            }
            spinor.setCplx(psi_c.CompC[0], c);
            MSG_INFO("sssspinor norm after linear_combination = " << spinor.norm());  // or getSquareNorm() on whichever component is populated
            psi_c.CompC[0] = nullptr; // ownership transferred to spinor, avoid double free
            spinor.calcSquareNorm();
        }
        // Placeholder columns must not enter the set: X = sum_i |phiS_i><phiL_i| pairs the large and
        // small sets by index, and the large (N_AO) and small (N_AO_small) AO counts differ.
        if (empty_spinor) continue;
        spinors.push_back(spinor);
    }
}

std::shared_ptr<mrcpp::CompFunctionVector> project_large_spinor_set(const Nuclei &nucs, const std::vector<std::string> &bas_files, const std::vector<std::string> &coef_files, double proj_prec, double screen, double coeff_thrs) {
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
std::shared_ptr<mrcpp::CompFunctionVector> project_small_spinor_set(const Nuclei &nucs, const std::vector<std::string> &large_bas_files, const std::vector<std::string> &coef_files, double proj_prec, double screen, double coeff_thrs) {
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

/** @brief constructor from  MW trees
 */
ASCOperator::ASCOperator(const Nuclei &nucs, const std::vector<std::string> &large_tree_paths, const std::vector<std::string> &small_tree_paths, double proj_prec, double screen, double coeff_thrs) {
    // Timer timer;
    
    auto spinors = std::make_shared<mrcpp::CompFunctionVector>(0);
    this->large = std::make_shared<mrcpp::CompFunctionVector>(0);
    this->small = std::make_shared<mrcpp::CompFunctionVector>(0);
    std::vector<ComplexDouble> imag1(2);
    imag1[0] = {1.0, 0.0};
    imag1[1] = {0.0, 1.0}; //{{1.0,0.0}, {0.0, 1.0}}; //1, i
    for (int i=0; i<large_tree_paths.size(); i++) {
        // std::string nuc_sym = nucs[0].getSymbol();
        std::vector<mrcpp::CompFunction<3>> large_comps(0);
        mrcpp::CompFunction<3> large_real;
        large_real.defreal();
        large_real.alloc(2, true);
        MSG_INFO("path_comp[0]="<<large_tree_paths[i]+"_Large_alpha_real");
        large_real.CompD[0]->loadTree(large_tree_paths[i]+"_Large_alpha_real");
        MSG_INFO("a "<< large_real.CompD[0]->getNNodes());
        large_real.CompD[1]->loadTree(large_tree_paths[i]+"_Large_beta_real");
        MSG_INFO("a1 "<< (large_real.CompD[1]->getNNodes()));
        // large_alpha_real.CompD[0]
        large_comps.push_back(large_real);

        MSG_INFO("b");
        mrcpp::CompFunction<3> large_imag;
        large_imag.defreal();
        large_imag.alloc(2, true);
        large_imag.CompD[0]->loadTree(large_tree_paths[i]+"_Large_alpha_imag");
        large_imag.CompD[1]->loadTree(large_tree_paths[i]+"_Large_beta_imag");
        large_comps.push_back(large_imag);
        mrcpp::CompFunction<3> large_tmp;
        large_tmp.defcomplex();
        large_tmp.alloc(2, true);
        mrcpp::linear_combination(large_tmp, imag1, large_comps, proj_prec, false);

        std::vector<mrcpp::CompFunction<3>> small_comps(0);
        MSG_INFO("c");
        this->large->push_back(large_tmp);
        MSG_INFO("d large ok path_small=" << small_tree_paths[i]+"_Small_alpha_real");
        MSG_INFO("d large ok path_Small=" << small_tree_paths[i]+"_Small_beta_real");
        mrcpp::CompFunction<3> small_real;
        small_real.defreal();
        small_real.alloc(2, true);
        small_real.CompD[0]->loadTree(small_tree_paths[i]+"_Small_alpha_real");
        small_real.CompD[1]->loadTree(small_tree_paths[i]+"_Small_beta_real");
        small_comps.push_back(small_real);
        // small_alpha_real.CompD[0]
        MSG_INFO("e");
        mrcpp::CompFunction<3> small_imag;
        small_imag.defreal();
        small_imag.alloc(2, true);
        small_imag.CompD[0]->loadTree(small_tree_paths[i]+"_Small_alpha_imag");
        small_imag.CompD[1]->loadTree(small_tree_paths[i]+"_Small_beta_imag");
        small_comps.push_back(small_imag);
        mrcpp::CompFunction<3> small_tmp;
        MSG_INFO("f");
        small_tmp.defcomplex();
        small_tmp.alloc(2, true);
        mrcpp::linear_combination(small_tmp, imag1, small_comps, proj_prec,false);
        this->small->push_back(small_tmp);
        MSG_INFO("g end");
    }
    //orthogonalising the large component between themselves
    ComplexMatrix SL = mrcpp::calc_overlap_matrix(*(this->large));
    ComplexMatrix U = math_utils::hermitian_matrix_pow(SL, -1.0);
    mrcpp::rotate(*(this->large), U, proj_prec);

    // mrcpp::print::time(2, "Gaussian coupling operator (large component, N=" + std::to_string(this->large->size()) + ")", timer);
    // mrcpp::print::time(2, "Gaussian coupling operator (small component, N=" + std::to_string(this->small->size()) + ")", timer);
}

/** @brief constructor from Gaussian bases. For now, it expects the output of the python scripts held in MRCHEM/python/DIRAC-to-trees_scripts.
 */
ASCOperator::ASCOperator(const Nuclei &nucs, const std::vector<std::string> &large_bas_files, const std::vector<std::string> &large_coef_files, const std::vector<std::string> &small_coef_files, double proj_prec, double screen, double coeff_thrs) {
    Timer timer;
    this->large = project_large_spinor_set(nucs, large_bas_files, large_coef_files, proj_prec, screen, coeff_thrs);
    this->small = project_small_spinor_set(nucs, large_bas_files, small_coef_files, proj_prec, screen, coeff_thrs);

    //orthogonalising the large component between themselves
    ComplexMatrix SL = mrcpp::calc_overlap_matrix(*(this->large));
    ComplexMatrix U = math_utils::hermitian_matrix_pow(SL, -1.0);
    mrcpp::rotate(*(this->large), U, proj_prec);

    if (large->size() != small->size()) MSG_ABORT("Large and small component size mismatch! Nbr of Large=" << large->size() << ", Nbr of small="<< small->size());

    // //debug
    // //===================================
    // std::vector<ComplexDouble> imag1(2);
    // imag1[0] = {1.0, 0.0};
    // imag1[1] = {0.0, 1.0}; //{{1.0,0.0}, {0.0, 1.0}}; //1, i
    // std::vector<mrcpp::CompFunction<3>> large_comps(0);
    // mrcpp::CompFunction<3> large_real;
    // large_real.defreal();
    // large_real.alloc(2, true);
    // MSG_INFO("path_comp[0]="<<"/home/qpitto/DIRAC_runs/ReMRChem/Runs/H2/H2_Large_alpha_real");
    // large_real.CompD[0]->loadTree("/home/qpitto/DIRAC_runs/ReMRChem/Runs/H2/H2_Large_alpha_real");
    // MSG_INFO("a "<< large_real.CompD[0]->getNNodes());
    // large_real.CompD[1]->loadTree("/home/qpitto/DIRAC_runs/ReMRChem/Runs/H2/H2_Large_beta_real");
    // MSG_INFO("a1 "<< (large_real.CompD[1]->getNNodes()));
    // // large_alpha_real.CompD[0]
    // large_comps.push_back(large_real);

    // MSG_INFO("b");
    // mrcpp::CompFunction<3> large_imag;
    // large_imag.defreal();
    // large_imag.alloc(2, true);
    // large_imag.CompD[0]->loadTree("/home/qpitto/DIRAC_runs/ReMRChem/Runs/H2/H2_Large_alpha_imag");
    // large_imag.CompD[1]->loadTree("/home/qpitto/DIRAC_runs/ReMRChem/Runs/H2/H2_Large_beta_imag");
    // large_comps.push_back(large_imag);
    // mrcpp::CompFunction<3> large_tmp;
    // large_tmp.defcomplex();
    // large_tmp.alloc(2, true);
    // mrcpp::linear_combination(large_tmp, imag1, large_comps, proj_prec, false);

    // std::vector<mrcpp::CompFunction<3>> small_comps(0);
    // MSG_INFO("c");
    // this->large->push_back(large_tmp);
    // MSG_INFO("d large ok path_small=" << small_tree_paths[i]+"_Small_alpha_real");
    // MSG_INFO("d large ok path_Small=" << small_tree_paths[i]+"_Small_beta_real");
    // mrcpp::CompFunction<3> small_real;
    // small_real.defreal();
    // small_real.alloc(2, true);
    // small_real.CompD[0]->loadTree(small_tree_paths[i]+"_Small_alpha_real");
    // small_real.CompD[1]->loadTree(small_tree_paths[i]+"_Small_beta_real");
    // small_comps.push_back(small_real);
    // // small_alpha_real.CompD[0]
    // MSG_INFO("e");
    // mrcpp::CompFunction<3> small_imag;
    // small_imag.defreal();
    // small_imag.alloc(2, true);
    // small_imag.CompD[0]->loadTree(small_tree_paths[i]+"_Small_alpha_imag");
    // small_imag.CompD[1]->loadTree(small_tree_paths[i]+"_Small_beta_imag");
    // small_comps.push_back(small_imag);
    // mrcpp::CompFunction<3> small_tmp;
    // MSG_INFO("f");
    // small_tmp.defcomplex();
    // small_tmp.alloc(2, true);
    // mrcpp::linear_combination(small_tmp, imag1, small_comps, proj_prec,false);
    // this->small->push_back(small_tmp);
    // MSG_INFO("g end");

    // //?===================================

    mrcpp::print::time(2, "Gaussian coupling operator (large component, N=" + std::to_string(this->large->size()) + ")", timer);
    mrcpp::print::time(2, "Gaussian coupling operator (small component, N=" + std::to_string(this->small->size()) + ")", timer);
}

OrbitalVector ASCOperator::operator()(OrbitalVector &inp) {
    // <phi^L_i|ket> matrix
    ComplexMatrix matrix_Lket = mrcpp::calc_overlap_matrix(*(this->large), inp); //N_AO x N matrix
    OrbitalVector out(0);
    for (int i=0; i<inp.size(); i++) {
        if (!mrcpp::mpi::my_func(i)) continue;
        auto col_Lket = matrix_Lket.col(i); //not an std::vector<ComplexDouble>, is some Eigen block instead, need to transmute
        std::vector<ComplexDouble> vec_Lket(col_Lket.begin(), col_Lket.end()); //transmuting the block to the needed type
        Orbital out_tmp (inp[i].getFuncData());
        mrcpp::linear_combination(out_tmp, vec_Lket, *(this->small),-1.0, false);
        out_tmp.func_ptr->data.n1[0] = inp[i].func_ptr->data.n1[0]; //transmitting the spin to out_tmp
        out.push_back(out_tmp);
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

    // MSG_INFO("left_mat= "<< left_matrix);
    // MSG_INFO("right_mat= "<< right_matrix);

    // ComplexMatrix left_matrix_debug = mrcpp::calc_overlap_matrix( *(this->small), *(this->small)); //N x N_AO matrix
    // //<phi^L_i|ket>
    // ComplexMatrix right_matrix_debug = mrcpp::calc_overlap_matrix(*(this->large), *(this->large)); //N_AO x N matrix
    // MSG_INFO("left_mat= "<< left_matrix_debug);
    // MSG_INFO("right_mat= "<< right_matrix_debug);

    std::stringstream o_name;
    o_name << "<i|" << this->name() << "|j>";
    // mrcpp::print::tree(2, o_name.str(), orbital::get_n_nodes(Oket), orbital::get_size_nodes(Oket), t1.elapsed());
    return left_matrix*right_matrix;
}

ComplexDouble ASCOperator::trace(OrbitalVector &bra, OrbitalVector &ket) {
    Timer t1;
    //NOTE: there might be a super smart way of avoiding to compute the full matrices but I don't see it right now
    //<bra|phi^S_i>
    ComplexMatrix left_matrix = mrcpp::calc_overlap_matrix(bra, *(this->small)); //N x N_AO matrix
    //<phi^L_i|ket>
    ComplexMatrix right_matrix = mrcpp::calc_overlap_matrix(*(this->large), ket); //N_AO x N matrix

    //Compute the trace
    ComplexDouble out = (left_matrix.array() * right_matrix.transpose().array()).sum();

    // std::stringstream o_name;
    // o_name << "<i|" << this->name() << "|j>";
    // mrcpp::print::tree(2, o_name.str(), orbital::get_n_nodes(Oket), orbital::get_size_nodes(Oket), t1.elapsed());
    return out;
}

} // namespace mrchem
