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

#include "MRCPP/Printer"

#include "HydrogenFunction.h"
#include "chemistry/PhysicalConstants.h"
#include "mrchem.h"
#include "utils/math_utils.h"

namespace mrchem {

RadialFunction::RadialFunction(int n, int l, double Z)
        : N(n)
        , L(l) {
    this->c_0 = calcConstant(Z);
    this->c_1 = 2.0 * Z / (1.0 * this->N);
    // if (N+L <9){ //sanity test
    //     if (N + L < 9) {
    //     double calcVer = this->calcConstant(Z) * this->evalfPoly(rho);
    //     double ref  = RadialFunction::calcConstant_hardcoded(Z) * RadialFunction::evalfPoly_hardcoded(rho);
    //     if (std::abs(calcVer - ref) > 1e-10 * std::max(std::abs(ref), 1.0)) MSG_ABORT("Generalised formula doesn't hold with hardcoded tests, computed constant="<< calcConstant(Z) << ", ref. value=" <<RadialFunction::calcConstant_hardcoded(Z));
    // }
}

double RadialFunction::evalf(const mrcpp::Coord<1> &r) const {
    double rho = this->c_1 * r[0];

    // //sanity tests
    // if (N+L<9) {
    //     if (std::abs(this->evalfPoly(rho)- RadialFunction::evalfPoly_hardcoded(rho))>1e-10 ) {
    //         MSG_ABORT("Generalised formula doesn't hold with hardcoded tests, computed constant="<< this->evalfPoly(rho) << ", ref. value=" <<RadialFunction::evalfPoly_hardcoded(rho));
    //     }
    // }
    return this->c_0 * this->evalfPoly(rho) * exp(-rho / 2.0);
}

// Copied from mrcpp::Gaussian, but should be visible at higher scales than Gaussians
bool RadialFunction::isVisibleAtScale(int scale, int nQuadPts) const {
    auto stdDev = calcStdDev();
    auto visibleScale = static_cast<int>(-std::floor(std::log2(nQuadPts * 5.0 * stdDev)));
    return (scale >= visibleScale);
}

// clang-format off
// double RadialFunction::calcConstant(double Z) const {
//     // R_NL(r) = c_0 * rho^L * L_{N-L-1}^{(2L+1)}(rho) * exp(-rho/2),  rho = c_1*r = 2*Z*r/N
//     // c_0 = Z^{3/2} * sqrt( (2/N)^3 / (2*N*(N+L)!*(N-L-1)!) )
//     double c2 = std::pow(2.0 / N, 3) / (2.0 * N * math_utils::factorial(N + L) * math_utils::factorial(N - L - 1));
//     return std::sqrt(c2) * std::pow(Z, 3.0 / 2.0);
// }
double RadialFunction::calcConstant(double Z) const {
    // c_0 = Z^{3/2} * sqrt( (2/N)^3 * (N-L-1)! / (2*N*(N+L)!) )
    double c2 = std::pow(2.0 / N, 3) * math_utils::factorial(N - L - 1) / (2.0 * N * math_utils::factorial(N + L));
    return std::sqrt(c2) * std::pow(Z, 3.0 / 2.0);
}
// clang-format on

// clang-format off
double RadialFunction::evalfPoly(double rho) const {
    // Returns rho^L * L_{N-L-1}^{(2L+1)}(rho), generalized Laguerre polynomial via the standard
    // 3-term recurrence (DLMF 18.9.13):
    //   L_0^a(x) = 1,  L_1^a(x) = 1 + a - x
    //   (k+1) L_{k+1}^a(x) = (2k+1+a-x) L_k^a(x) - (k+a) L_{k-1}^a(x)
    int kmax = this->N - this->L - 1;
    int a = 2 * this->L + 1;
    double Lkm1 = 1.0;             // L_0^a(rho)
    double Lk = 1.0 + a - rho;     // L_1^a(rho)
    double lag = Lkm1;
    for (int k = 1; k <= kmax - 1; k++) {
        double Lk1 = ((2 * k + 1 + a - rho) * Lk - (k + a) * Lkm1) / (k + 1);
        Lkm1 = Lk;
        Lk = Lk1;
    }
    if (kmax >= 1) lag = Lk;
    return std::pow(rho, this->L) * lag;
}
// clang-format on

// clang-format off
double RadialFunction::calcConstant_hardcoded(double Z) const {
    double c = 0.0;
    if (N == 1 and L == 0) { c = 2.0;
    } else if (N == 2 and L == 0) { c = 1.0/(   2.0*std::sqrt(  2.0));
    } else if (N == 3 and L == 0) { c = 1.0/(   9.0*std::sqrt(  3.0));
    } else if (N == 4 and L == 0) { c = 1.0/(  48.0*std::sqrt(  4.0));
    } else if (N == 5 and L == 0) { c = 1.0/( 300.0*std::sqrt(  5.0));
    } else if (N == 6 and L == 0) { c = 1.0/(2160.0*std::sqrt(  6.0));
    } else if (N == 2 and L == 1) { c = 1.0/(   2.0*std::sqrt(  6.0));
    } else if (N == 3 and L == 1) { c = 1.0/(   9.0*std::sqrt(  6.0));
    } else if (N == 4 and L == 1) { c = 1.0/(  32.0*std::sqrt( 15.0));
    } else if (N == 5 and L == 1) { c = 1.0/( 150.0*std::sqrt( 30.0));
    } else if (N == 6 and L == 1) { c = 1.0/( 432.0*std::sqrt(210.0));
    } else if (N == 3 and L == 2) { c = 1.0/(   9.0*std::sqrt( 30.0));
    } else if (N == 4 and L == 2) { c = 1.0/(  96.0*std::sqrt(  5.0));
    } else if (N == 5 and L == 2) { c = 1.0/( 150.0*std::sqrt( 70.0));
    } else if (N == 6 and L == 2) { c = 1.0/( 864.0*std::sqrt(105.0));
    } else if (N == 4 and L == 3) { c = 1.0/(  96.0*std::sqrt( 35.0));
    } else if (N == 5 and L == 3) { c = 1.0/( 300.0*std::sqrt( 70.0));
    } else { NOT_IMPLEMENTED_ABORT; }
    return c*pow(Z, 3.0/2.0);
}
// clang-format on

// clang-format off
double RadialFunction::evalfPoly_hardcoded(double r) const {
    double value = 0.0;
    if (N == 1 and L == 0) { value =        (  1.0                                                                     );
    } else if (N == 2 and L == 0) { value =        (  2.0 -    1.0*r                                                          );
    } else if (N == 3 and L == 0) { value =        (  6.0 -    6.0*r +    1.0*r*r                                             );
    } else if (N == 4 and L == 0) { value =        ( 24.0 -   36.0*r +   12.0*r*r -   1.0*r*r*r                               );
    } else if (N == 5 and L == 0) { value =        (120.0 -  240.0*r +  120.0*r*r -  20.0*r*r*r +  1.0*r*r*r*r                );
    } else if (N == 6 and L == 0) { value =        (720.0 - 1800.0*r + 1200.0*r*r - 300.0*r*r*r + 30.0*r*r*r*r - 1.0*r*r*r*r*r);
    } else if (N == 2 and L == 1) { value =      r*(  1.0                                                                     );
    } else if (N == 3 and L == 1) { value =      r*(  4.0 -    1.0*r                                                          );
    } else if (N == 4 and L == 1) { value =      r*( 20.0 -   10.0*r +    1.0*r*r                                             );
    } else if (N == 5 and L == 1) { value =      r*(120.0 -   90.0*r +   18.0*r*r -   1.0*r*r*r                               );
    } else if (N == 6 and L == 1) { value =      r*(840.0 -  840.0*r +  252.0*r*r -  28.0*r*r*r +  1.0*r*r*r*r                );
    } else if (N == 3 and L == 2) { value =    r*r*(  1.0                                                                     );
    } else if (N == 4 and L == 2) { value =    r*r*(  6.0 -    1.0*r                                                          );
    } else if (N == 5 and L == 2) { value =    r*r*( 42.0 -   14.0*r +    1.0*r*r                                             );
    } else if (N == 6 and L == 2) { value =    r*r*(336.0 -  168.0*r +   24.0*r*r -   1.0*r*r*r                               );
    } else if (N == 4 and L == 3) { value =  r*r*r*(  1.0                                                                     );
    } else if (N == 5 and L == 3) { value =  r*r*r*(  8.0 -    1.0*r                                                          );
    } else { NOT_IMPLEMENTED_ABORT; }
    return value;
}
// clang-format on

AngularFunction::AngularFunction(int l, int m)
        : L(l)
        , M(m) {
    this->c_0 = calcConstant();
}

double AngularFunction::evalf(const mrcpp::Coord<3> &r) const {
    return this->c_0 * this->evalfPoly(r);
}

// clang-format off
double AngularFunction::calcConstant() const {
    double c = 0.0;
    double sqrt_pi = std::sqrt(mrcpp::pi);
    if (L == 0 and M == 0) { c = std::sqrt(  1.0/1.0);
    } else if (L == 1 and M == 0) { c = std::sqrt(  3.0/1.0);
    } else if (L == 1 and M == 1) { c = std::sqrt(  3.0/1.0);
    } else if (L == 1 and M == 2) { c = std::sqrt(  3.0/1.0);
    } else if (L == 2 and M == 0) { c = std::sqrt( 60.0/4.0);
    } else if (L == 2 and M == 1) { c = std::sqrt( 60.0/4.0);
    } else if (L == 2 and M == 2) { c = std::sqrt( 60.0/4.0);
    } else if (L == 2 and M == 3) { c = std::sqrt( 15.0/4.0);
    } else if (L == 2 and M == 4) { c = std::sqrt(  5.0/4.0);
    } else if (L == 3 and M == 0) { c = std::sqrt(  7.0/4.0);
    } else if (L == 3 and M == 1) { c = std::sqrt(  7.0/4.0);
    } else if (L == 3 and M == 2) { c = std::sqrt(  7.0/4.0);
    } else if (L == 3 and M == 3) { c = std::sqrt(105.0/4.0);
    } else if (L == 3 and M == 4) { c = std::sqrt(105.0/4.0);
    } else if (L == 3 and M == 5) { c = std::sqrt(105.0/4.0);
    } else if (L == 3 and M == 6) { c = std::sqrt(105.0/4.0);
    } else { NOT_IMPLEMENTED_ABORT; }
    return c/(2.0*sqrt_pi);
}
// clang-format on

// clang-format off
double AngularFunction::evalfPoly(const mrcpp::Coord<3> &q) const {
    double x = q[0];
    double y = q[1];
    double z = q[2];
    double r = std::sqrt(x*x + y*y + z*z);
    double value = 0.0;
    if (L == 0 and M == 0) { value = 1.0;
    } else if (L == 1 and M == 0) { value = x/r;
    } else if (L == 1 and M == 1) { value = y/r;
    } else if (L == 1 and M == 2) { value = z/r;
    } else if (L == 2 and M == 0) { value = x*y/(r*r);
    } else if (L == 2 and M == 1) { value = x*z/(r*r);
    } else if (L == 2 and M == 2) { value = y*z/(r*r);
    } else if (L == 2 and M == 3) { value = (x*x - y*y)/(r*r);
    } else if (L == 2 and M == 4) { value = (2.0*z*z - x*x - y*y)/(r*r);
    } else if (L == 3 and M == 0) { value = x*(5.0*x*x-3.0*r*r)/(r*r*r);
    } else if (L == 3 and M == 1) { value = y*(5.0*y*y-3.0*r*r)/(r*r*r);
    } else if (L == 3 and M == 2) { value = z*(5.0*z*z-3.0*r*r)/(r*r*r);
    } else if (L == 3 and M == 3) { value = x*(z*z - y*y)/(r*r*r);
    } else if (L == 3 and M == 4) { value = y*(z*z - x*x)/(r*r*r);
    } else if (L == 3 and M == 5) { value = z*(x*x - y*y)/(r*r*r);
    } else if (L == 3 and M == 6) { value = 2.0*x*y*z/(r*r*r);
    } else { NOT_IMPLEMENTED_ABORT; }
    return value;
}
// clang-format on

HydrogenFunction::HydrogenFunction(int n, int l, int m, double Z)
        : RepresentableFunction<3>()
        , origin({0.0, 0.0, 0.0})
        , R(n, l, Z)
        , Y(l, m) {}

HydrogenFunction::HydrogenFunction(int n, int l, int m, double Z, const mrcpp::Coord<3> &o)
        : RepresentableFunction<3>()
        , origin(o)
        , R(n, l, Z)
        , Y(l, m) {}

double HydrogenFunction::evalf(const mrcpp::Coord<3> &p) const {
    const mrcpp::Coord<3> &o = this->origin;
    mrcpp::Coord<3> q{p[0] - o[0], p[1] - o[1], p[2] - o[2]};
    mrcpp::Coord<1> r{math_utils::calc_distance(p, o)};
    return this->R.evalf(r) * this->Y.evalf(q);
}

// Copied from mrcpp::Gaussian
bool HydrogenFunction::isZeroOnInterval(const double *a, const double *b) const {
    bool outside = true;
    double stdDev = this->R.calcStdDev();
    for (int i = 0; i < 3; i++) {
        double gaussBoxMin = this->origin[i] - 15.0 * stdDev;
        double gaussBoxMax = this->origin[i] + 15.0 * stdDev;
        outside &= (a[i] < gaussBoxMax and b[i] > gaussBoxMin);
    }
    return not(outside);
}

} // namespace mrchem
