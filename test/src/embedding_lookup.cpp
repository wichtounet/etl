//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "bias_test.hpp"

// Tests for embedding_lookup

TEMPLATE_TEST_CASE_2("embedding_lookup/0", "[embedding_lookup]", T, float, double) {
    etl::fast_matrix<T, 6> a({1, 2, 3, 2, 6, 0});
    etl::fast_matrix<T, 8, 3> b{ 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2,  0, 0, 0,  2, 1, 3 };

    etl::fast_matrix<T, 6, 3> c;

    c = embedding_lookup(a, b);

    REQUIRE_EQUALS(c(0, 0), T(4));
    REQUIRE_EQUALS(c(0, 1), T(5));
    REQUIRE_EQUALS(c(0, 2), T(6));

    REQUIRE_EQUALS(c(1, 0), T(0.1));
    REQUIRE_EQUALS(c(1, 1), T(0.2));
    REQUIRE_EQUALS(c(1, 2), T(0.3));

    REQUIRE_EQUALS(c(2, 0), T(-1.0));
    REQUIRE_EQUALS(c(2, 1), T(-1.0));
    REQUIRE_EQUALS(c(2, 2), T(0.0));

    REQUIRE_EQUALS(c(3, 0), T(0.1));
    REQUIRE_EQUALS(c(3, 1), T(0.2));
    REQUIRE_EQUALS(c(3, 2), T(0.3));

    REQUIRE_EQUALS(c(4, 0), T(0));
    REQUIRE_EQUALS(c(4, 1), T(0));
    REQUIRE_EQUALS(c(4, 2), T(0));

    REQUIRE_EQUALS(c(5, 0), T(1));
    REQUIRE_EQUALS(c(5, 1), T(2));
    REQUIRE_EQUALS(c(5, 2), T(3));
}

TEMPLATE_TEST_CASE_2("embedding_lookup/1", "[embedding_lookup]", T, float, double) {
    etl::fast_matrix<T, 6> a({1, 2, 3, 2, 6, 0});
    etl::fast_matrix<T, 8, 3> b{ 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2,  0, 0, 0,  2, 1, 3 };

    etl::fast_matrix<T, 6, 3> c;

    c = T(1.1);
    c += embedding_lookup(a, b);

    REQUIRE_EQUALS(c(0, 0), T(1.1) + T(4));
    REQUIRE_EQUALS(c(0, 1), T(1.1) + T(5));
    REQUIRE_EQUALS(c(0, 2), T(1.1) + T(6));

    REQUIRE_EQUALS(c(1, 0), T(1.1) + T(0.1));
    REQUIRE_EQUALS(c(1, 1), T(1.1) + T(0.2));
    REQUIRE_EQUALS(c(1, 2), T(1.1) + T(0.3));

    REQUIRE_EQUALS(c(2, 0), T(1.1) + T(-1.0));
    REQUIRE_EQUALS(c(2, 1), T(1.1) + T(-1.0));
    REQUIRE_EQUALS(c(2, 2), T(1.1) + T(0.0));

    REQUIRE_EQUALS(c(3, 0), T(1.1) + T(0.1));
    REQUIRE_EQUALS(c(3, 1), T(1.1) + T(0.2));
    REQUIRE_EQUALS(c(3, 2), T(1.1) + T(0.3));

    REQUIRE_EQUALS(c(4, 0), T(1.1) + T(0));
    REQUIRE_EQUALS(c(4, 1), T(1.1) + T(0));
    REQUIRE_EQUALS(c(4, 2), T(1.1) + T(0));

    REQUIRE_EQUALS(c(5, 0), T(1.1) + T(1));
    REQUIRE_EQUALS(c(5, 1), T(1.1) + T(2));
    REQUIRE_EQUALS(c(5, 2), T(1.1) + T(3));
}

TEMPLATE_TEST_CASE_2("batch_embedding_lookup/0", "[batch_embedding_lookup]", T, float, double) {
    etl::fast_matrix<T, 2, 6> a({1, 2, 3, 2, 6, 0, 1, 3, 2, 2, 6, 0});
    etl::fast_matrix<T, 8, 3> b{ 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2,  0, 0, 0,  2, 1, 3 };

    etl::fast_matrix<T, 2, 6, 3> c;

    c = batch_embedding_lookup(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), T(4));
    REQUIRE_EQUALS(c(0, 0, 1), T(5));
    REQUIRE_EQUALS(c(0, 0, 2), T(6));

    REQUIRE_EQUALS(c(0, 1, 0), T(0.1));
    REQUIRE_EQUALS(c(0, 1, 1), T(0.2));
    REQUIRE_EQUALS(c(0, 1, 2), T(0.3));

    REQUIRE_EQUALS(c(0, 2, 0), T(-1.0));
    REQUIRE_EQUALS(c(0, 2, 1), T(-1.0));
    REQUIRE_EQUALS(c(0, 2, 2), T(0.0));

    REQUIRE_EQUALS(c(0, 3, 0), T(0.1));
    REQUIRE_EQUALS(c(0, 3, 1), T(0.2));
    REQUIRE_EQUALS(c(0, 3, 2), T(0.3));

    REQUIRE_EQUALS(c(0, 4, 0), T(0));
    REQUIRE_EQUALS(c(0, 4, 1), T(0));
    REQUIRE_EQUALS(c(0, 4, 2), T(0));

    REQUIRE_EQUALS(c(0, 5, 0), T(1));
    REQUIRE_EQUALS(c(0, 5, 1), T(2));
    REQUIRE_EQUALS(c(0, 5, 2), T(3));

    REQUIRE_EQUALS(c(1, 0, 0), T(4));
    REQUIRE_EQUALS(c(1, 0, 1), T(5));
    REQUIRE_EQUALS(c(1, 0, 2), T(6));

    REQUIRE_EQUALS(c(1, 1, 0), T(-1.0));
    REQUIRE_EQUALS(c(1, 1, 1), T(-1.0));
    REQUIRE_EQUALS(c(1, 1, 2), T(0.0));

    REQUIRE_EQUALS(c(1, 2, 0), T(0.1));
    REQUIRE_EQUALS(c(1, 2, 1), T(0.2));
    REQUIRE_EQUALS(c(1, 2, 2), T(0.3));

    REQUIRE_EQUALS(c(1, 3, 0), T(0.1));
    REQUIRE_EQUALS(c(1, 3, 1), T(0.2));
    REQUIRE_EQUALS(c(1, 3, 2), T(0.3));

    REQUIRE_EQUALS(c(1, 4, 0), T(0));
    REQUIRE_EQUALS(c(1, 4, 1), T(0));
    REQUIRE_EQUALS(c(1, 4, 2), T(0));

    REQUIRE_EQUALS(c(1, 5, 0), T(1));
    REQUIRE_EQUALS(c(1, 5, 1), T(2));
    REQUIRE_EQUALS(c(1, 5, 2), T(3));
}

TEMPLATE_TEST_CASE_2("embedding_gradients/0", "[embedding_gradients]", T, float, double) {
    etl::fast_matrix<T, 6> a({1, 2, 3, 2, 6, 0});
    etl::fast_matrix<T, 6, 3> b{ 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2 };

    etl::fast_matrix<T, 8, 3> c;
    c = embedding_gradients(a, b, c);

    REQUIRE_EQUALS(c(0, 0), T(2));
    REQUIRE_EQUALS(c(0, 1), T(2));
    REQUIRE_EQUALS(c(0, 2), T(2));

    REQUIRE_EQUALS(c(1, 0), T(1));
    REQUIRE_EQUALS(c(1, 1), T(2));
    REQUIRE_EQUALS(c(1, 2), T(3));

    REQUIRE_EQUALS(c(2, 0), T(3.0));
    REQUIRE_EQUALS(c(2, 1), T(4.0));
    REQUIRE_EQUALS(c(2, 2), T(6.0));

    REQUIRE_EQUALS(c(3, 0), T(0.1));
    REQUIRE_EQUALS(c(3, 1), T(0.2));
    REQUIRE_EQUALS(c(3, 2), T(0.3));

    REQUIRE_EQUALS(c(4, 0), T(0));
    REQUIRE_EQUALS(c(4, 1), T(0));
    REQUIRE_EQUALS(c(4, 2), T(0));

    REQUIRE_EQUALS(c(5, 0), T(0));
    REQUIRE_EQUALS(c(5, 1), T(0));
    REQUIRE_EQUALS(c(5, 2), T(0));

    REQUIRE_EQUALS(c(6, 0), 1);
    REQUIRE_EQUALS(c(6, 1), 1);
    REQUIRE_EQUALS(c(6, 2), 1);

    REQUIRE_EQUALS(c(7, 0), 0);
    REQUIRE_EQUALS(c(7, 1), 0);
    REQUIRE_EQUALS(c(7, 2), 0);
}

TEMPLATE_TEST_CASE_2("batch_embedding_gradients/0", "[batch_embedding_gradients]", T, float, double) {
    etl::fast_matrix<T, 2, 6> a({1, 2, 3, 2, 6, 0, 1, 2, 3, 2, 6, 0});
    etl::fast_matrix<T, 2, 6, 3> b{ 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2, 1, 2, 3,  4, 5, 6,  0.1, 0.2, 0.3,  -1.0, -1.0, 0.0,  1, 1, 1,  2, 2, 2 };

    etl::fast_matrix<T, 8, 3> c;
    c = batch_embedding_gradients(a, b, c);

    REQUIRE_EQUALS(c(0, 0), T(2) * T(2));
    REQUIRE_EQUALS(c(0, 1), T(2) * T(2));
    REQUIRE_EQUALS(c(0, 2), T(2) * T(2));

    REQUIRE_EQUALS(c(1, 0), T(2) * T(1));
    REQUIRE_EQUALS(c(1, 1), T(2) * T(2));
    REQUIRE_EQUALS(c(1, 2), T(2) * T(3));

    REQUIRE_EQUALS(c(2, 0), T(2) * T(3.0));
    REQUIRE_EQUALS(c(2, 1), T(2) * T(4.0));
    REQUIRE_EQUALS(c(2, 2), T(2) * T(6.0));

    REQUIRE_EQUALS(c(3, 0), T(2) * T(0.1));
    REQUIRE_EQUALS(c(3, 1), T(2) * T(0.2));
    REQUIRE_EQUALS(c(3, 2), T(2) * T(0.3));

    REQUIRE_EQUALS(c(4, 0), T(0));
    REQUIRE_EQUALS(c(4, 1), T(0));
    REQUIRE_EQUALS(c(4, 2), T(0));

    REQUIRE_EQUALS(c(5, 0), T(0));
    REQUIRE_EQUALS(c(5, 1), T(0));
    REQUIRE_EQUALS(c(5, 2), T(0));

    REQUIRE_EQUALS(c(6, 0), T(2) * T(1));
    REQUIRE_EQUALS(c(6, 1), T(2) * T(1));
    REQUIRE_EQUALS(c(6, 2), T(2) * T(1));

    REQUIRE_EQUALS(c(7, 0), 0);
    REQUIRE_EQUALS(c(7, 1), 0);
    REQUIRE_EQUALS(c(7, 2), 0);
}
