//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

ETL_TEST_CASE("conv/4d/full/mixed/0", "[conv][conv4][full]") {
    etl::fast_matrix<float, 10, 12, 5, 5> I;
    etl::fast_matrix<double, 12, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<float, 10, 2, 7, 7> ref;
    etl::fast_matrix<float, 10, 2, 7, 7> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
                }
            }
        }
    }

    c = conv_4d_full(I, K);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], base_eps * 100000);
    }
}

ETL_TEST_CASE("conv/4d/full/mixed/1", "[conv][conv4][full]") {
    etl::fast_matrix<float, 10, 12, 5, 5> I;
    etl::fast_matrix_cm<float, 12, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<float, 10, 2, 7, 7> ref;
    etl::fast_matrix<float, 10, 2, 7, 7> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
                }
            }
        }
    }

    c = conv_4d_full(I, K);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], base_eps * 100000);
    }
}
