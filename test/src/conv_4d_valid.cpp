//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

CONV4_VALID_TEST_CASE("conv_4d/valid_1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 6, 2, 17, 17> I;
    etl::fast_matrix<T, 7, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 6, 7, 15, 15> ref;
    etl::fast_matrix<T, 6, 7, 15, 15> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_VALID_TEST_CASE("conv_4d/valid_2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 7, 4, 15, 15> I;
    etl::fast_matrix<T, 8, 4, 5, 5> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 7, 8, 11, 11> ref;
    etl::fast_matrix<T, 7, 8, 11, 11> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv_4d/valid_3", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 4, 22, 22> I;
    etl::fast_matrix<T, 2, 4, 8, 8> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 5, 2, 15, 15> ref;
    etl::fast_matrix<T, 5, 2, 15, 15> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid_flipped(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv_4d/valid_4", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 4, 23, 21> I;
    etl::fast_matrix<T, 2, 4, 9, 7> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 5, 2, 15, 15> ref;
    etl::fast_matrix<T, 5, 2, 15, 15> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid_flipped(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv_4d/valid_5", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 4, 23, 24> I;
    etl::fast_matrix<T, 2, 4, 15, 16> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 5, 2, 9, 9> ref;
    etl::fast_matrix<T, 5, 2, 9, 9> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid_flipped(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv_4d/valid_6", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 4, 3, 27, 27> I;
    etl::fast_matrix<T, 2, 3, 19, 19> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 4, 2, 9, 9> ref;
    etl::fast_matrix<T, 4, 2, 9, 9> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid_flipped(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}
