//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_4d_valid

CONV4_VALID_TEST_CASE("conv_4d/valid_1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 2, 5, 5> I(etl::sequence_generator(10.0) * 4.0);
    etl::fast_matrix<T, 12, 2, 3, 3> K(etl::sequence_generator(2.0) * 0.3);

    etl::fast_matrix<T, 10, 12, 3, 3> ref;
    etl::fast_matrix<T, 10, 12, 3, 3> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(k) += conv_2d_valid(I(i)(c), K(k)(c));
            }
        }
    }

    Impl::apply(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE(c[i] == Approx(ref[i]));
    }
}

// conv_4d_full

CONV4_FULL_TEST_CASE("conv_4d/full_1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 12, 5, 5> I(etl::sequence_generator(10.0) * 4.0);
    etl::fast_matrix<T, 12, 2, 3, 3> K(etl::sequence_generator(2.0) * 0.3);

    etl::fast_matrix<T, 10, 2, 7, 7> ref;
    etl::fast_matrix<T, 10, 2, 7, 7> c;

    ref = 0.0;
    for(std::size_t i = 0; i < etl::dim<0>(I); ++i){
        for(std::size_t c = 0; c < etl::dim<1>(K); ++c){
            for(std::size_t k = 0; k < etl::dim<0>(K); ++k){
                ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
            }
        }
    }

    Impl::apply(I, K, c);

    for(std::size_t i = 0; i < ref.size(); ++i){
        REQUIRE(c[i] == Approx(ref[i]));
    }
}
