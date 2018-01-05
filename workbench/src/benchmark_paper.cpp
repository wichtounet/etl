//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "etl/etl.hpp"

#define CPM_BENCHMARK "Paper Benchmarks"
#define CPM_NO_RANDOMIZATION            // Randomly initialize only once
#define CPM_AUTO_STEPS                  // Enable steps estimation system
#define CPM_STEP_ESTIMATION_MIN 0.1     // Run during 0.1 seconds for estimating steps
#define CPM_RUNTIME_TARGET 2.0          // Run each test during 1.0 seconds

#ifdef ETL_EGBLAS_MODE
#define CPM_PROLOGUE cudaDeviceSynchronize();
#endif

#include "cpm/cpm.hpp"

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;
using smat3 = etl::dyn_matrix<float, 3>;
using smat4 = etl::dyn_matrix<float, 4>;

using paper_vector_policy = VALUES_POLICY(10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000);

using sgemm_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200));

using neural_net_policy= VALUES_POLICY(10, 20, 30, 40, 50, 60, 70, 80, 90, 100);


//Bench saxpy
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        paper_vector_policy,
        "saxpy [paper][axpy][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& x, svec& y, svec& yy){ yy = 3.3f * x + y; },
        [](size_t d){ return 2 * d; }
        );
}

//Bench saxpy_saxpy
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        paper_vector_policy,
        "saxpy_saxpy [paper][axpy][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& x, svec& y, svec& yy){ yy = (2.2f * y + x) >> (3.3f * x + y); },
        [](size_t d){ return 2 * d; }
        );
}

//Bench sapxmbpy
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        paper_vector_policy,
        "sapxmbpy [paper][axpy][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& x, svec& y, svec& yy){ yy = (3.3f + x) >> (4.4f + y); },
        [](size_t d){ return 3 * d; }
        );
}

//Bench C = A * B
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        sgemm_policy,
        "sgemm [paper][gemm][s]",
        [](size_t d1, size_t d2){ return std::make_tuple(smat(d1, d2), smat(d1, d2), smat(d1, d2)); },
        [](smat& x, smat& y, smat& yy){ yy = x * y; },
        [](size_t d1, size_t d2){ return d1 * d2 * d1; }
        );
}

//Bench C = A * (A * 1.2 + C) * (-1.2 * B - A)
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        sgemm_policy,
        "complex_sgemm [paper][gemm][s]",
        [](size_t d1, size_t d2){ return std::make_tuple(smat(d1, d2), smat(d1, d2), smat(d1, d2)); },
        [](smat& x, smat& y, smat& yy){ yy = x * (x * 1.2f + y) * (-1.2f * y - x); },
        [](size_t d1, size_t d2){ return 3 * d1 * d2 * d1 + 4 * d1 * d2; }
        );
}

//Bench neural net
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        neural_net_policy,
        "neural_net [paper][gemm][s]",
        [](size_t d1){ return std::make_tuple(smat(d1, 784), smat(d1, 1000), smat(d1, 1000), smat(d1, 10), smat(784, 1000), smat(1000, 1000), smat(1000, 10), svec(1000), svec(1000), svec(10)); },
        [](smat& i1, smat& o1, smat& o2, smat& o3, smat& w1, smat& w2, smat& w3, svec& b1, svec& b2, svec& b3){
            o1 = etl::sigmoid(bias_add_2d(i1 * w1, b1));
            o2 = etl::sigmoid(bias_add_2d(o1 * w2, b2));
            o3 = etl::sigmoid(bias_add_2d(o2 * w3, b3));
        },
        [](size_t d1){ return d1 * d1 * d1 * 3; }
        );
}
