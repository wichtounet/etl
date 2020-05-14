//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_hidden [rbm]",
    [](size_t d1, size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d2), dvec(d2), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& b, dvec& t, dmat& w){ h = etl::sigmoid(b + etl::mul(v, w, t)); }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_visible [rbm]",
    [](size_t d1, size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1), dvec(d1), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& c, dvec& t, dmat& w){ v = etl::sigmoid(c + etl::mul(w, h, t)); }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_hidden [crbm]",
    [](size_t nc, size_t k, size_t nv, size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dmat3(nc,nv,nv), dmat3(k, nh, nh), dmat4(nc, k, nh, nh));},
    [](dmat4& w, dvec& b, dmat3& v, dmat3& h, dmat4& v_cv){
        for (size_t k = 0; k < etl::dim<0>(w); ++k) {
            v_cv(k) = conv_2d_valid_multi_flipped(v(k), w(k));
        }

        auto b_rep = etl::force_temporary(etl::rep(b, etl::dim<1>(h), etl::dim<2>(h)));
        h = etl::sigmoid(b_rep + etl::sum_l(v_cv));
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(
        VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40),        // NC
        VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40),  // K
        VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36),  // NV
        VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),  // NH
    "conv_rbm_visible [crbm]",
    [](size_t nc, size_t k, size_t nv, size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(nc), dmat3(nc,nv,nv), dmat3(k,nh,nh), dmat3(k,nv,nv));},
    [](dmat4& w, dvec& c, dmat3& v, dmat3& h, dmat3& h_cv){
        for(size_t channel = 0; channel < etl::dim<0>(c); ++channel){
            for(size_t k = 0; k < etl::dim<1>(w); ++k){
                h_cv(k) = etl::conv_2d_full(h(k), w(channel)(k));
            }

            v(channel) = sigmoid(c(channel) + sum_l(h_cv));
        }
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(
        VALUES_POLICY(1,  1,  1,  3,  3,  3,  20, 40),  // NC
        VALUES_POLICY(10, 10, 20, 20, 20, 30, 30, 30),  // K
        VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36),  // NV
        VALUES_POLICY(5,  11, 19, 19, 19, 19, 19, 19)), // NH
    "conv_rbm_hidden_batch_32 [crbm]",
    [](size_t nc, size_t k, size_t nv, size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dmat4(32UL,nc,nv,nv), dmat4(32UL,k, nh, nh), dmat5(32UL, nc, k, nh, nh));},
    [](dmat4& w, dvec& b, dmat4& v, dmat4& h, dmat5& v_cv){
        //Note dyn rep only handles two dim
        auto b_rep = etl::force_temporary(etl::rep(b, etl::dim<2>(h), etl::dim<3>(h)));

        for(size_t b = 0; b < 32UL; ++b){
            for (size_t k = 0; k < etl::dim<0>(w); ++k) {
                v_cv(b)(k) = conv_2d_valid_multi_flipped(v(b)(k), w(k));
            }

            h(b) = etl::sigmoid(b_rep + etl::sum_l(v_cv(b)));
        }
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_visible_batch_64 [crbm]",
    [](size_t nc, size_t k, size_t nv, size_t nh) {
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(nc), dmat4(64UL,nc,nv,nv), dmat4(64UL,k,nh,nh), dmat4(64UL,k,nv,nv)); },
    [](dmat4& w, dvec& c, dmat4& v, dmat4& h, dmat4& h_cv) {
        for (size_t b = 0; b < 64UL; ++b) {
            for (size_t channel = 0; channel < etl::dim<0>(c); ++channel) {
                for (size_t k = 0; k < etl::dim<1>(w); ++k) {
                    h_cv(b)(k) = etl::conv_2d_full(h(b)(k), w(channel)(k));
                }

                v(b)(channel) = sigmoid(c(channel) + sum_l(h_cv(b)));
            }
        }
    }
)
