//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

CONV4_VALID_FILTER_TEST_CASE("conv/4d/valid/filter/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(-1.0) * 0.01;
    K = etl::sequence_generator(-2.0) * 0.03;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += conv_2d_valid(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}

CONV4_VALID_FILTER_TEST_CASE("conv/4d/valid/filter/2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(-1.0) * 0.01;
    K = etl::sequence_generator(-2.0) * 0.03;

    etl::fast_matrix<T, 4, 3, 5, 5> ref;
    etl::fast_matrix<T, 4, 3, 5, 5> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += etl::conv_2d_valid<1, 1, 1, 1>(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::template apply<1, 1, 1, 1>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}

DYN_CONV4_VALID_FILTER_TEST_CASE("conv/4d/valid/filter/3", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(-1.0) * 0.01;
    K = etl::sequence_generator(-2.0) * 0.03;

    etl::fast_matrix<T, 4, 3, 5, 5> ref;
    etl::fast_matrix<T, 4, 3, 5, 5> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += etl::conv_2d_valid<1, 1, 1, 1>(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::apply(I, K, c, 1, 1, 1, 1);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}

CONV4_VALID_FILTER_TEST_CASE("conv/4d/valid/filter/4", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 8, 8> I;
    etl::fast_matrix<T, 10, 4, 5, 5> K;

    I = etl::sequence_generator(-1.0) * 0.01;
    K = etl::sequence_generator(-2.0) * 0.03;

    etl::fast_matrix<T, 4, 3, 6, 6> ref;
    etl::fast_matrix<T, 4, 3, 6, 6> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += etl::conv_2d_valid<1, 1, 1, 1>(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::template apply<1, 1, 1, 1>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}

CONV4_VALID_FILTER_FLIPPED_TEST_CASE("conv/4d/valid/filter/flipped/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.024;
    K = etl::sequence_generator(2.0) * 0.043;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += conv_2d_valid(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}
