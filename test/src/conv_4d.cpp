//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_4d_valid

CONV4_VALID_TEST_CASE("conv_4d/valid_1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 2, 5, 5> I;
    etl::fast_matrix<T, 12, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 12, 3, 3> ref;
    etl::fast_matrix<T, 10, 12, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_VALID_TEST_CASE("conv_4d/valid_2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 10, 4, 4> ref;
    etl::fast_matrix<T, 9, 10, 4, 4> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv_4d/valid_3", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 10, 4, 4> ref;
    etl::fast_matrix<T, 9, 10, 4, 4> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += conv_2d_valid(I(i)(c), fflip(K(k)(c)));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

// conv_4d_full

CONV4_FULL_TEST_CASE("conv/4d/full/1", "[conv][conv4][full]") {
    etl::fast_matrix<T, 10, 12, 5, 5> I;
    etl::fast_matrix<T, 12, 2, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.4;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 10, 2, 7, 7> ref;
    etl::fast_matrix<T, 10, 2, 7, 7> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], base_eps * 100000);
    }
}

CONV4_FULL_TEST_CASE("conv/4d/full/2", "[conv][conv4][full]") {
    etl::fast_matrix<T, 8, 9, 6, 6> I;
    etl::fast_matrix<T, 9, 3, 3, 3> K;

    I = etl::sequence_generator(2) * -0.3;
    K = etl::sequence_generator(-2.0) * 0.16;

    etl::fast_matrix<T, 8, 3, 8, 8> ref;
    etl::fast_matrix<T, 8, 3, 8, 8> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_FULL_TEST_CASE("conv/4d/full/3", "[conv][conv4][full]") {
    etl::fast_matrix<T, 5, 3, 6, 6> I;
    etl::fast_matrix<T, 3, 2, 3, 3> K;

    I = etl::sequence_generator(1.1) * -3.0;
    K = etl::sequence_generator(-1.3) * 1.6;

    etl::fast_matrix<T, 5, 2, 8, 8> ref;
    etl::fast_matrix<T, 5, 2, 8, 8> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), K(k)(c));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_FULL_FLIPPED_TEST_CASE("conv/4d/full/flipped/1", "[conv][conv4][full]") {
    etl::fast_matrix<T, 7, 4, 6, 6> I;
    etl::fast_matrix<T, 4, 2, 3, 3> K;

    I = etl::sequence_generator(15.0) * -3.0;
    K = etl::sequence_generator(-4.0) * 1.6;

    etl::fast_matrix<T, 7, 2, 8, 8> ref;
    etl::fast_matrix<T, 7, 2, 8, 8> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), fflip(K(k)(c)));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

CONV4_FULL_FLIPPED_TEST_CASE("conv/4d/full/flipped/2", "[conv][conv4][full]") {
    etl::fast_matrix<T, 5, 3, 6, 6> I;
    etl::fast_matrix<T, 3, 2, 3, 3> K;

    I = etl::sequence_generator(1.1) * -3.0;
    K = etl::sequence_generator(-1.3) * 1.6;

    etl::fast_matrix<T, 5, 2, 8, 8> ref;
    etl::fast_matrix<T, 5, 2, 8, 8> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (std::size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(c) += conv_2d_full(I(i)(k), fflip(K(k)(c)));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], ref[i]);
    }
}

// conv_4d_valid_filter

CONV4_VALID_FILTER_TEST_CASE("conv/4d/valid/filter/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(-1.0) * 0.01;
    K = etl::sequence_generator(-2.0) * 0.03;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (std::size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += conv_2d_valid(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}

// conv_4d_valid_filter_flipped

CONV4_VALID_FILTER_FLIPPED_TEST_CASE("conv/4d/valid/filter/flipped/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 10, 3, 5, 5> I;
    etl::fast_matrix<T, 10, 4, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.024;
    K = etl::sequence_generator(2.0) * 0.043;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (std::size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (std::size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (std::size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += conv_2d_valid_flipped(I(i)(c), fflip(K(i)(k)));
                }
            }
        }
    }

    Impl::apply(I, K, c);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.05);
    }
}
