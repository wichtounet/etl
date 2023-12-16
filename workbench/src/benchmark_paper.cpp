//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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

//Bench dense neural net
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        neural_net_policy,
        "dense_neural_net [paper][gemm][s]",
        [](size_t d1){ return std::make_tuple(smat(d1, 784), smat(d1, 1000), smat(d1, 1000), smat(d1, 10), smat(784, 1000), smat(1000, 1000), smat(1000, 10), svec(1000), svec(1000), svec(10)); },
        [](smat& i1, smat& o1, smat& o2, smat& o3, smat& w1, smat& w2, smat& w3, svec& b1, svec& b2, svec& b3){
            o1 = etl::sigmoid(bias_add_2d(i1 * w1, b1));
            o2 = etl::sigmoid(bias_add_2d(o1 * w2, b2));
            o3 = etl::sigmoid(bias_add_2d(o2 * w3, b3));
        },
        [](size_t d1){ return d1 * d1 * d1 * 3; }
        );
}

//Bench conv neural net
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        neural_net_policy,
        "conv_neural_net [paper][gemm][s]",
        [](size_t d1){ return std::make_tuple( smat4(d1, 3, 32, 32), smat4(d1, 16, 32, 32), smat4(d1, 16, 16, 16), smat4(d1, 32, 16, 16), smat4(d1, 32, 8, 8), smat(d1, 1000), smat(d1, 10), smat4(16, 3, 3, 3), smat4(32, 16, 3, 3), smat(32 * 8 * 8, 1000), smat(1000, 10), svec(16), svec(32), svec(1000), svec(10)); },
        [](smat4& i1, smat4& o1, smat4& o2, smat4& o3, smat4& o4, smat& o5, smat& o6, smat4& w1, smat4& w2, smat& w3, smat& w4, svec& b1, svec& b2, svec& b3, svec& b4){
            o1 = etl::ml::convolution_forward<1, 1, 1, 1>(i1, w1);
            o1 = bias_add_4d(o1, b1);
            o1 = relu(o1);

            o2 = etl::ml::max_pool_forward<2, 2>(o1);

            o3 = etl::ml::convolution_forward<1, 1, 1, 1>(o2, w2);
            o3 = bias_add_4d(o3, b2);
            o3 = relu(o3);

            o4 = etl::ml::max_pool_forward<2, 2>(o3);

            o5 = etl::relu(bias_add_2d(etl::reshape(o4, etl::dim<0>(i1), 32 * 8 * 8) * w3, b3));
            o6 = etl::sigmoid(bias_add_2d(o5 * w4, b4));
        },
        [](size_t d1){ return d1 * d1 * d1 * 3; }
        );
}

//Bench large conv neural net
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        neural_net_policy,
        "large_conv_neural_net [paper][gemm][s]",
        [](size_t d1){ return std::make_tuple( smat4(d1, 3, 256, 256), smat4(d1, 32, 256, 256), smat4(d1, 32, 128, 128), smat4(d1, 32, 128, 128), smat4(d1, 32, 64, 64), smat4(d1, 32, 64, 64), smat4(d1, 32, 32, 32),smat4(d1, 32, 32, 32), smat4(d1, 32, 16, 16),smat4(d1, 32, 16, 16), smat4(d1, 32, 8, 8),smat(d1, 2048),smat(d1, 1000),smat4(32, 3, 3, 3), smat4(32, 32, 3, 3), smat4(32, 32, 3, 3), smat4(32, 32, 3, 3), smat4(32, 32, 3, 3), smat(32 * 8 * 8, 2048), smat(2048, 1000), svec(32), svec(32), svec(32), svec(32), svec(32), svec(2048), svec(2000)); },
        [](smat4& i1, smat4& o1, smat4& o1_p, smat4& o2, smat4& o2_p, smat4& o3, smat4& o3_p, smat4& o4, smat4& o4_p, smat4& o5, smat4& o5_p, smat& o6, smat& o7, smat4& w1, smat4& w2, smat4& w3, smat4& w4, smat4& w5, smat& w6, smat& w7, svec& b1, svec& b2, svec& b3, svec& b4, svec& b5, svec& b6, svec& b7){

            o1 = etl::ml::convolution_forward<1, 1, 1, 1>(i1, w1);
            o1 = bias_add_4d(o1, b1);
            o1 = relu(o1);
            o1_p = etl::ml::max_pool_forward<2, 2>(o1);

            o2 = etl::ml::convolution_forward<1, 1, 1, 1>(o1_p, w2);
            o2 = bias_add_4d(o2, b2);
            o2 = relu(o2);
            o2_p = etl::ml::max_pool_forward<2, 2>(o2);

            o3 = etl::ml::convolution_forward<1, 1, 1, 1>(o2_p, w3);
            o3 = bias_add_4d(o3, b3);
            o3 = relu(o3);
            o3_p = etl::ml::max_pool_forward<2, 2>(o3);

            o4 = etl::ml::convolution_forward<1, 1, 1, 1>(o3_p, w4);
            o4 = bias_add_4d(o4, b4);
            o4 = relu(o4);
            o4_p = etl::ml::max_pool_forward<2, 2>(o4);

            o5 = etl::ml::convolution_forward<1, 1, 1, 1>(o4_p, w5);
            o5 = bias_add_4d(o5, b5);
            o5 = relu(o5);
            o5_p = etl::ml::max_pool_forward<2, 2>(o5);

            o6 = etl::relu(bias_add_2d(etl::reshape(o5_p, etl::dim<0>(i1), 32 * 8 * 8) * w6, b6));
            o7 = etl::sigmoid(bias_add_2d(o6 * w7, b7));
        },
        [](size_t d1){ return d1 * d1 * d1 * 3; }
        );
}
