//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_hidden [rbm]",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d2), dvec(d2), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& b, dvec& t, dmat& w){ h = etl::sigmoid(b + etl::mul(v, w, t)); }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_visible [rbm]",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1), dvec(d1), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& c, dvec& t, dmat& w){ v = etl::sigmoid(c + etl::mul(w, h, t)); }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_hidden [crbm]",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dmat3(nc,nv,nv), dmat3(k, nh, nh), dmat4(nc, k, nh, nh));},
    [](dmat4& w, dvec& b, dmat3& v, dmat3& h, dmat4& v_cv){
        conv_3d_valid_multi_flipped(v, w, v_cv);

        auto b_rep = etl::force_temporary(etl::rep(b, etl::dim<1>(h), etl::dim<2>(h)));
        h = etl::sigmoid(b_rep + etl::sum_l(v_cv));
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_visible [crbm]",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dvec(nc), dmat3(nc,nv,nv), dmat3(k,nh,nh), dmat3(k,nv,nv));},
    [](dmat4& w, dvec& b, dvec& c, dmat3& v, dmat3& h, dmat3& h_cv){
        for(std::size_t channel = 0; channel < etl::dim<0>(c); ++channel){
            for(std::size_t k = 0; k < etl::dim<0>(b); ++k){
                h_cv(k) = etl::conv_2d_full(h(k), w(channel)(k));
            }

            v(channel) = sigmoid(c(channel) + sum_l(h_cv));
        }
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_hidden_batch_64 [crbm]",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dmat4(64UL,nc,nv,nv), dmat4(64UL,k, nh, nh), dmat5(64UL, nc, k, nh, nh));},
    [](dmat4& w, dvec& b, dmat4& v, dmat4& h, dmat5& v_cv){
        //Note dyn rep only handles two dim
        auto b_rep = etl::force_temporary(etl::rep(b, etl::dim<1>(h), etl::dim<2>(h)));

        for(std::size_t b = 0; b < 64UL; ++b){
            conv_3d_valid_multi_flipped(v(b), w, v_cv(b));

            h(b) = etl::sigmoid(b_rep + etl::sum_l(v_cv(b)));
        }
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_visible_batch_64 [crbm]",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh) {
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dvec(nc), dmat4(64UL,nc,nv,nv), dmat4(64UL,k,nh,nh), dmat4(64UL,k,nv,nv)); },
    [](dmat4& w, dvec& b, dvec& c, dmat4& v, dmat4& h, dmat4& h_cv) {
        for (std::size_t batch = 0; batch < 64UL; ++batch) {
            for (std::size_t channel = 0; channel < etl::dim<0>(c); ++channel) {
                for (std::size_t k = 0; k < etl::dim<0>(b); ++k) {
                    h_cv(batch)(k) = etl::conv_2d_full(h(batch)(k), w(channel)(k));
                }

                v(batch)(channel) = sigmoid(c(channel) + sum_l(h_cv(batch)));
            }
        }
    }
)
