//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_4d_valid

CONV4_VALID_TEST_CASE("conv/4d/stride/valid_1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 2, 9, 9> I(etl::sequence_generator(10.0) * 4.0);
    etl::fast_matrix<T, 12, 2, 3, 3> K(etl::sequence_generator(2.0) * 0.3);

    etl::fast_matrix<T, 10, 12, 4, 4> ref;
    etl::fast_matrix<T, 10, 12, 4, 4> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(k) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(k)(c));
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_VALID_TEST_CASE("conv/4d/stride/valid_2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 4, 4> I(etl::sequence_generator(-10.0) * 0.04);
    etl::fast_matrix<T, 10, 4, 2, 2> K(etl::sequence_generator(-2.0) * 1.56);

    etl::fast_matrix<T, 9, 10, 2, 2> ref;
    etl::fast_matrix<T, 9, 10, 2, 2> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(k) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(k)(c));
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

// conv_4d_valid_flipped

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 6, 6> I(etl::sequence_generator(-10.0) * 0.04);
    etl::fast_matrix<T, 10, 4, 2, 2> K(etl::sequence_generator(-2.0) * 1.56);

    etl::fast_matrix<T, 9, 10, 3, 3> ref;
    etl::fast_matrix<T, 9, 10, 3, 3> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(k) += etl::conv_2d_valid_flipped<2,2,0,0>(I(i)(c), K(k)(c));
            }
        }
    }

    Impl::template apply<2,2,0,0>(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 5, 5> I(etl::sequence_generator(-10.0) * 0.04);
    etl::fast_matrix<T, 10, 4, 3, 3> K(etl::sequence_generator(-2.0) * 1.56);

    etl::fast_matrix<T, 9, 10, 2, 2> ref;
    etl::fast_matrix<T, 9, 10, 2, 2> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(k) += etl::conv_2d_valid_flipped<2, 2, 0, 0>(I(i)(c), K(k)(c));
            }
        }
    }

    Impl::template apply<2,2,0,0>(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}
