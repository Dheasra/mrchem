#ifndef MWMULTIPLIER_H
#define MWMULTIPLIER_H

#include "TreeBuilder.h"
#include "MultiplicationCalculator.h"
#include "GridGenerator.h"
#include "WaveletAdaptor.h"

template<int D>
class MWMultiplier : public TreeBuilder<D> {
public:
    MWMultiplier(const MultiResolutionAnalysis<D> &mra, double pr = -1.0)
            : TreeBuilder<D>(mra),
              prec(pr) {
    }
    virtual ~MWMultiplier() {
    }

    void setPrecision(double pr) { this->prec = pr; }
    void multPrecision(double fac) { this->prec *= fac; }

    FunctionTree<D>* operator()(FunctionTreeVector<D> &inp) {
        FunctionTree<D> *out = new FunctionTree<D>(this->MRA);

        Timer init_t;
        init_t.restart();
        GridGenerator<D> G(this->MRA);
        G(*out, inp);
        init_t.stop();
        println(10, "Time initializing   " << init_t);

        (*this)(*out, inp);
        return out;
    }

    void operator()(FunctionTree<D> &out,
                    FunctionTreeVector<D> &inp,
                    int maxIter = -1) {
        this->adaptor = new WaveletAdaptor<D>(this->prec, this->MRA.getMaxScale());
        this->calculator = new MultiplicationCalculator<D>(inp);
        this->build(out, maxIter);
        this->clearCalculator();
        this->clearAdaptor();

        Timer trans_t;
        trans_t.restart();
        out.mwTransform(BottomUp);
        out.calcSquareNorm();
        trans_t.stop();

        Timer clean_t;
        clean_t.restart();
        for (int i = 0; i < inp.size(); i++) {
            FunctionTree<D> &tree = inp.getFunc(i);
            tree.deleteGenerated();
        }
        clean_t.stop();

        println(10, "Time transform      " << trans_t);
        println(10, "Time cleaning       " << clean_t);
        println(10, std::endl);
    }

protected:
    double prec;
};

#endif // MWMULTIPLIER_H