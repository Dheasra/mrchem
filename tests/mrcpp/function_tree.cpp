#include "catch.hpp"

#include "factory_functions.h"

namespace function_tree {

template<int D> void testZeroFunction();
template<int D> void testGeneratedNodes();

SCENARIO("Zero FunctionTree", "[function_tree_zero], [function_tree], [trees]") {
    GIVEN("a default function in 1D") {
        testZeroFunction<1>();
    }
    GIVEN("a default function in 2D") {
        testZeroFunction<2>();
    }
    GIVEN("a default function in 3D") {
        testZeroFunction<3>();
    }
}

template<int D> void testZeroFunction() {
    MultiResolutionAnalysis<D> *mra = 0;
    initialize(&mra);
    GridGenerator<D> G(*mra);
    finalize(&mra);

    FunctionTree<D> *tree = G();
    WHEN("the function is set to zero") {
        tree->setZero();
        THEN("its value in an arbitrary point is zero") {
            double r[3] = {-0.2, 0.6, 0.76};
            REQUIRE( tree->evalf(r) == Approx(0.0) );
        }
        THEN("its squared norm is zero") {
            REQUIRE( tree->getSquareNorm() == Approx(0.0) );
        }
        THEN("it integrates to zero") {
            REQUIRE( tree->integrate() == Approx(0.0) );
        }
        THEN("the dot product with itself is zero") {
            REQUIRE( tree->dot(*tree) == Approx(0.0) );
        }
    }
    delete tree;
}

SCENARIO("Generating FunctionTree nodes", "[function_tree_generating], [function_tree], [trees]") {
    GIVEN("a default function in 1D") {
        testGeneratedNodes<1>();
    }
    GIVEN("a default function in 2D") {
        testGeneratedNodes<2>();
    }
    GIVEN("a default function in 3D") {
        testGeneratedNodes<3>();
    }
}

template<int D> void testGeneratedNodes() {
    const double r[3] = {-0.3, 0.6, 1.9};
    const int depth = 3;

    MultiResolutionAnalysis<D> *mra = 0;
    initialize(&mra);
    GridGenerator<D> G(*mra);
    finalize(&mra);

    FunctionTree<D> *tree = G();
    tree->setZero();

    THEN("there are no GenNodes") {
        REQUIRE( tree->getNGenNodes() == 0 );
        REQUIRE( tree->getNAllocGenNodes() == 0 );
    }

    WHEN("a non-existing node is fetched") {
        MWNode<D> &node = tree->getNode(r, depth);

        THEN("there will be allocated GenNodes") {
            REQUIRE( tree->getNGenNodes() > 0 );
            REQUIRE( tree->getNAllocGenNodes() > 0 );

            AND_WHEN("the GenNodes are cleared") {
                tree->clearGenerated();

                THEN("there will be un-allocated GenNodes") {
                    REQUIRE( tree->getNGenNodes() > 0 );
                    REQUIRE( tree->getNAllocGenNodes() == 0 );
                }
            }

            AND_WHEN("the GenNodes are deleted") {
                tree->deleteGenerated();
                THEN("there will be no GenNodes") {
                    REQUIRE( tree->getNGenNodes() == 0 );
                    REQUIRE( tree->getNAllocGenNodes() == 0 );
                }
            }
        }
    }
    delete tree;
}

} // namespace