//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_4d_valid

CONV4_VALID_TEST_CASE("conv/4d/stride/valid/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 2, 9, 9> I;
    etl::fast_matrix<T, 6, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 5, 6, 4, 4> ref;
    etl::fast_matrix<T, 5, 6, 4, 4> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_TEST_CASE("conv/4d/stride/valid/2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 4, 4> I;
    etl::fast_matrix<T, 5, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 5, 2, 2> ref;
    etl::fast_matrix<T, 9, 5, 2, 2> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_TEST_CASE("conv/4d/stride/valid/3", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 4, 4> I;
    etl::fast_matrix<T, 5, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 5, 5, 5> ref;
    etl::fast_matrix<T, 9, 5, 5, 5> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid<1, 1, 1, 1>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<1, 1, 1, 1>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_TEST_CASE("conv/4d/stride/valid/4", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 2, 9, 9> I;
    etl::fast_matrix<T, 6, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 5, 6, 11, 11> ref;
    etl::fast_matrix<T, 5, 6, 11, 11> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid<1, 1, 2, 2>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<1, 1, 2, 2>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

DYN_CONV4_VALID_TEST_CASE("conv/4d/stride/valid/5", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 2, 9, 9> I;
    etl::fast_matrix<T, 6, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 5, 6, 4, 4> ref;
    etl::fast_matrix<T, 5, 6, 4, 4> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply(I, K, c, 2, 2, 0, 0);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

// conv_4d_valid_flipped

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 6, 6> I;
    etl::fast_matrix<T, 5, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 5, 3, 3> ref;
    etl::fast_matrix<T, 9, 5, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid_flipped<2, 2, 0, 0>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/2", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 5, 5> I;
    etl::fast_matrix<T, 5, 4, 3, 3> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 5, 2, 2> ref;
    etl::fast_matrix<T, 9, 5, 2, 2> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid_flipped<2, 2, 0, 0>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/3", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 9, 4, 4, 4> I;
    etl::fast_matrix<T, 5, 4, 2, 2> K;

    I = etl::sequence_generator(-10.0) * 0.04;
    K = etl::sequence_generator(-2.0) * 1.56;

    etl::fast_matrix<T, 9, 5, 5, 5> ref;
    etl::fast_matrix<T, 9, 5, 5, 5> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid_flipped<1, 1, 1, 1>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<1, 1, 1, 1>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

CONV4_VALID_FLIPPED_TEST_CASE("conv/4d/stride/valid/flipped/4", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 2, 9, 9> I;
    etl::fast_matrix<T, 6, 2, 3, 3> K;

    I = etl::sequence_generator(10.0) * 4.0;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 5, 6, 11, 11> ref;
    etl::fast_matrix<T, 5, 6, 11, 11> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t c = 0; c < etl::dim<1>(K); ++c) {
                for (size_t k = 0; k < etl::dim<0>(K); ++k) {
                    ref(i)(k) += etl::conv_2d_valid_flipped<1, 1, 2, 2>(I(i)(c), K(k)(c));
                }
            }
        }
    }

    Impl::template apply<1, 1, 2, 2>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

// conv_4d_valid_filter

CONV4_VALID_FILTER_TEST_CASE("conv/4d/stride/valid/filter/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 3, 7, 7> I;
    etl::fast_matrix<T, 5, 4, 3, 3> K;

    I = etl::sequence_generator(-1.0) * 0.0019;
    K = etl::sequence_generator(-2.0) * 0.0023;

    etl::fast_matrix<T, 4, 3, 3, 3> ref;
    etl::fast_matrix<T, 4, 3, 3, 3> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}

// conv_4d_valid_filter_flipped

CONV4_VALID_FILTER_FLIPPED_TEST_CASE("conv/4d/stride/valid/filter/flipped/1", "[conv][conv4][valid]") {
    etl::fast_matrix<T, 5, 3, 5, 5> I;
    etl::fast_matrix<T, 5, 4, 3, 3> K;

    I = etl::sequence_generator(3.0) * 0.04;
    K = etl::sequence_generator(2.0) * 0.3;

    etl::fast_matrix<T, 4, 3, 2, 2> ref;
    etl::fast_matrix<T, 4, 3, 2, 2> c;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref = 0.0;
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            for (size_t k = 0; k < etl::dim<1>(K); ++k) {
                for (size_t c = 0; c < etl::dim<1>(I); ++c) {
                    ref(k)(c) += etl::conv_2d_valid<2, 2, 0, 0>(I(i)(c), K(i)(k));
                }
            }
        }
    }

    Impl::template apply<2, 2, 0, 0>(I, K, c);

    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], ref[i], 0.1);
    }
}
