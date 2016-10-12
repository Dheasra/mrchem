//#include <fstream>

#include "Orbital.h"
#include "FunctionTree.h"

using namespace std;

Orbital::Orbital(int occ, int s)
        : spin(s),
          occupancy(occ),
          error(1.0),
          real(0),
          imag(0) {
}

Orbital::Orbital(const Orbital &orb)
        : spin(orb.spin),
          occupancy(orb.occupancy),
          error(1.0),
          real(0),
          imag(0) {
}

Orbital& Orbital::operator=(const Orbital &orb) {
    if (this != &orb) {
        if (this->real != 0) MSG_ERROR("Orbital not empty");
        if (this->imag != 0) MSG_ERROR("Orbital not empty");
        this->spin = orb.spin;
        this->occupancy = orb.occupancy;
        this->error = orb.error;
        this->real = orb.real;
        this->imag = orb.imag;
    }
    return *this;
}

void Orbital::clear(bool free) {
    if (this->real != 0 and free) delete this->real;
    if (this->imag != 0 and free) delete this->imag;
    this->real = 0;
    this->imag = 0;
}

int Orbital::getNNodes() const {
    int nNodes = 0;
    if (this->real != 0) nNodes += this->real->getNNodes();
    if (this->imag != 0) nNodes += this->imag->getNNodes();
    return nNodes;
}

double Orbital::getSquareNorm() const {
    double sqNorm = 0.0;
    if (this->real != 0) sqNorm += this->real->getSquareNorm();
    if (this->imag != 0) sqNorm += this->imag->getSquareNorm();
    return sqNorm;
}

void Orbital::compare(const Orbital &orb) const {
    if (this->compareOccupancy(orb) < 0) {
        MSG_WARN("Different occupancy");
    }
    if (this->compareSpin(orb) < 0) {
        MSG_WARN("Different spin");
    }
}

int Orbital::compareOccupancy(const Orbital &orb) const {
    if (this->getOccupancy() == orb.getOccupancy()) {
        return this->getOccupancy();
    }
    return -1;
}

int Orbital::compareSpin(const Orbital &orb) const {
    if (this->getSpin() == orb.getSpin()) {
        return this->getSpin();
    }
    return -1;
}

double Orbital::getExchangeFactor(const Orbital &orb) const {
    if (orb.getSpin() == Paired) {
        return 1.0;
    } else if (this->getSpin() == Paired) {
        return 0.5;
    } else if (this->getSpin() == orb.getSpin()) {
        return 1.0;
    }
    return 0.0;
}

bool Orbital::isConverged(double prec) const {
    if (getError() > prec) {
        return false;
    } else {
        return true;
    }
}

complex<double> Orbital::dot(Orbital &ket) {
    Orbital &bra = *this;
    if ((bra.getSpin() == Alpha) and (ket.getSpin() == Beta)) {
        return 0.0;
    }
    if ((bra.getSpin() == Beta) and (ket.getSpin() == Alpha)) {
        return 0.0;
    }
    double re = 0.0;
    double im = 0.0;
    if (bra.hasReal() and ket.hasReal()) re += bra.re().dot(ket.re());
    if (bra.hasImag() and ket.hasImag()) re += bra.im().dot(ket.im());
    if (bra.hasReal() and ket.hasImag()) im += bra.re().dot(ket.im());
    if (bra.hasImag() and ket.hasReal()) im -= bra.im().dot(ket.re());
    return complex<double>(re, im);
}

void Orbital::normalize() {
    double norm = sqrt(getSquareNorm());
    if (hasReal()) this->re() *= 1.0/norm;
    if (hasImag()) this->im() *= 1.0/norm;
}

void Orbital::orthogonalize(Orbital &phi) {
    NOT_IMPLEMENTED_ABORT;
}

