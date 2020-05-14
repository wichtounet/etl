//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

TEMPLATE_TEST_CASE_2("conv/4d/backward/0", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 5, 5> I;
    etl::fast_matrix<T, 8, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 7, 7> ref;
    etl::fast_matrix<T, 10, 2, 7, 7> c;

    ref = etl::conv_4d_full(I, K);
    c = etl::conv_4d_backward<1,1,0,0>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward/1", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 5, 5> I;
    etl::fast_matrix<T, 8, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 5, 5> ref;
    etl::fast_matrix<T, 10, 2, 5, 5> c;

    ref = etl::conv_4d_valid_back<1,1,1,1>(I, K);
    c = etl::conv_4d_backward<1,1,1,1>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, 10.0 * base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward/2", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 7, 7> I;
    etl::fast_matrix<T, 8, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 5, 5> ref;
    etl::fast_matrix<T, 10, 2, 5, 5> c;

    c = etl::conv_4d_backward<1,1,2,2>(I, K);
    ref = etl::conv_4d_valid_back<1,1,0,0>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, 10.0 * base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward/3", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 16, 16> I;
    etl::fast_matrix<T, 8, 2, 7, 7> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 12, 12> ref;
    etl::fast_matrix<T, 10, 2, 12, 12> c;

    c = etl::conv_4d_backward<1,1,5,5>(I, K);
    ref = etl::conv_4d_valid_back<1,1,1,1>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, 10.0 * base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward/4", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 4, 4> I;
    etl::fast_matrix<T, 8, 2, 5, 5> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 11, 11> ref;
    etl::fast_matrix<T, 10, 2, 11, 11> c;

    c = etl::conv_4d_backward<2,2,0,0>(I, K);

    auto I_s = etl::impl::common::inner_pad(I, 2, 2);
    ref = etl::conv_4d_full(I_s, K);

    REQUIRE_DIRECT(approx_equals(c, ref, 10.0 * base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward/5", "[conv][conv4][backward]", T, float, double) {
    etl::fast_matrix<T, 10, 8, 4, 4> I;
    etl::fast_matrix<T, 8, 2, 5, 5> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 9, 9> ref;
    etl::fast_matrix<T, 10, 2, 9, 9> c;

    c = etl::conv_4d_backward<2,2,1,1>(I, K);

    auto I_s = etl::impl::common::inner_pad(I, 2, 2);
    ref = etl::conv_4d_valid_back<1, 1, 3, 3>(I_s, K);

    REQUIRE_DIRECT(approx_equals(c, ref, 10.0 * base_eps_etl));
}
