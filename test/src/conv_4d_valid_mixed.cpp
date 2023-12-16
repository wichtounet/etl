//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

ETL_TEST_CASE("conv4/valid/mixed/0", "[conv][conv4][valid]") {
    etl::fast_matrix<float, 6, 2, 17, 17> I;
    etl::fast_matrix<double, 7, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<float, 6, 7, 15, 15> ref;
    etl::fast_matrix<float, 6, 7, 15, 15> c;

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

    c = etl::conv_4d_valid(I, K);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

ETL_TEST_CASE("conv4/valid/mixed/1", "[conv][conv4][valid]") {
    etl::fast_matrix<float, 6, 2, 17, 17> I;
    etl::fast_matrix_cm<float, 7, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<float, 6, 7, 15, 15> ref;
    etl::fast_matrix<float, 6, 7, 15, 15> c;

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

    c = etl::conv_4d_valid(I, K);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}
