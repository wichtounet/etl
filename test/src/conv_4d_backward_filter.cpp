//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

TEMPLATE_TEST_CASE_2("conv/4d/backward_filter/0", "[conv][conv4][backward_filter]", T, float, double) {
    etl::fast_matrix<T, 10, 3, 6, 6> I;
    etl::fast_matrix<T, 10, 4, 4, 4> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    ref = etl::conv_4d_valid_filter<1, 1, 0, 0>(I, K);
    c = etl::conv_4d_backward_filter<1, 1, 0, 0>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward_filter/1", "[conv][conv4][backward_filter]", T, float, double) {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 5, 5> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    ref = etl::conv_4d_valid_filter<1, 1, 1, 1>(I, K);
    c = etl::conv_4d_backward_filter<1, 1, 1, 1>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward_filter/2", "[conv][conv4][backward_filter]", T, float, double) {
    etl::fast_matrix<T, 10, 3, 6, 6> I;
    etl::fast_matrix<T, 10, 4, 6, 6> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    ref = etl::conv_4d_valid_filter<1, 1, 1, 1>(I, K);
    c = etl::conv_4d_backward_filter<1, 1, 1, 1>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward_filter/3", "[conv][conv4][backward_filter]", T, float, double) {
    etl::fast_matrix<T, 10, 3, 7, 7> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    auto SK = etl::impl::common::inner_pad(K, 2, 2);
    ref = etl::conv_4d_valid_filter<1, 1, 0, 0>(I, SK);
    c = etl::conv_4d_backward_filter<2, 2, 0, 0>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps));
}

TEMPLATE_TEST_CASE_2("conv/4d/backward_filter/4", "[conv][conv4][backward_filter]", T, float, double) {
    etl::fast_matrix<T, 10, 3, 7, 7> I;
    etl::fast_matrix<T, 10, 4, 4, 4> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    auto SK = etl::impl::common::inner_pad(K, 2, 2);
    ref = etl::conv_4d_valid_filter<1, 1, 1, 1>(I, SK);
    c = etl::conv_4d_backward_filter<2, 2, 1, 1>(I, K);

    REQUIRE_DIRECT(approx_equals(c, ref, base_eps));
}
