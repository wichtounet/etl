//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * Suite of tests to make sure that CPU and GPU are correctly
 * updated when operations are mixed.
 *
 * It is necessary to use large matrices in orders to ensure that
 * GPU is used.
 */


#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

#ifdef ETL_CUDA

#ifdef ETL_EGBLAS_MODE

TEMPLATE_TEST_CASE_2("gpu/status/0", "[gpu]", T, float, double) {
    etl::fast_matrix<T, 75, 75> a;
    etl::fast_matrix<T, 75, 75> b;
    etl::fast_matrix<T, 75, 75> c;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();

    c = -(a + b);

    REQUIRE(c.is_gpu_up_to_date());
}

TEMPLATE_TEST_CASE_2("gpu/status/1", "[gpu]", T, float, double) {
    etl::fast_matrix<T, 75, 75> a;
    etl::fast_matrix<T, 75, 75> b;
    etl::fast_matrix<T, 75, 75> c;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();

    c = exp(a + b) >> log(a - b);

    REQUIRE(c.is_gpu_up_to_date());
}

#endif

TEMPLATE_TEST_CASE_2("gpu/1", "[gpu]", T, float, double) {
    etl::fast_matrix<T, 200, 200> a;
    etl::fast_matrix<T, 200, 200> b;

    a = 1;
    b = 2;

    etl::fast_matrix<T, 200, 200> c1;
    etl::fast_matrix<T, 200, 200> c2;

    c1 = a * b;
    c2 = c1;

    REQUIRE_EQUALS(c1(0, 0), 400);
    REQUIRE_EQUALS(c1(0, 1), 400);
    REQUIRE_EQUALS(c1(0, 2), 400);
    REQUIRE_EQUALS(c1(1, 0), 400);
    REQUIRE_EQUALS(c1(1, 1), 400);
    REQUIRE_EQUALS(c1(1, 2), 400);
    REQUIRE_EQUALS(c1(2, 0), 400);
    REQUIRE_EQUALS(c1(2, 1), 400);
    REQUIRE_EQUALS(c1(2, 2), 400);

    REQUIRE_EQUALS(c2(0, 0), 400);
    REQUIRE_EQUALS(c2(0, 1), 400);
    REQUIRE_EQUALS(c2(0, 2), 400);
    REQUIRE_EQUALS(c2(1, 0), 400);
    REQUIRE_EQUALS(c2(1, 1), 400);
    REQUIRE_EQUALS(c2(1, 2), 400);
    REQUIRE_EQUALS(c2(2, 0), 400);
    REQUIRE_EQUALS(c2(2, 1), 400);
    REQUIRE_EQUALS(c2(2, 2), 400);
}

TEMPLATE_TEST_CASE_2("gpu/2", "[gpu]", T, float, double) {
    etl::fast_matrix<T, 200, 200> a;
    etl::fast_matrix<T, 200, 200> b;

    a = 1;
    b = 2;

    etl::fast_matrix<T, 200, 200> c1;
    etl::fast_matrix<T, 200, 200> c2;
    c2 = 1;

    c1 = a * b;
    c2 += c1;

    REQUIRE_EQUALS(c1(0, 0), 400);
    REQUIRE_EQUALS(c1(0, 1), 400);
    REQUIRE_EQUALS(c1(0, 2), 400);
    REQUIRE_EQUALS(c1(1, 0), 400);
    REQUIRE_EQUALS(c1(1, 1), 400);
    REQUIRE_EQUALS(c1(1, 2), 400);
    REQUIRE_EQUALS(c1(2, 0), 400);
    REQUIRE_EQUALS(c1(2, 1), 400);
    REQUIRE_EQUALS(c1(2, 2), 400);

    REQUIRE_EQUALS(c2(0, 0), 401);
    REQUIRE_EQUALS(c2(0, 1), 401);
    REQUIRE_EQUALS(c2(0, 2), 401);
    REQUIRE_EQUALS(c2(1, 0), 401);
    REQUIRE_EQUALS(c2(1, 1), 401);
    REQUIRE_EQUALS(c2(1, 2), 401);
    REQUIRE_EQUALS(c2(2, 0), 401);
    REQUIRE_EQUALS(c2(2, 1), 401);
    REQUIRE_EQUALS(c2(2, 2), 401);
}

TEMPLATE_TEST_CASE_2("gpu/3", "[gpu]", T, float, double) {
    etl::fast_matrix<T, 200, 200> a;
    etl::fast_matrix<T, 200, 200> b;

    a = 1;
    b = 2;

    etl::fast_matrix<T, 200, 200> c1;

    c1 = a * b;
    c1 *= 2.0;

    REQUIRE_EQUALS(c1(0, 0), 800);
    REQUIRE_EQUALS(c1(0, 1), 800);
    REQUIRE_EQUALS(c1(0, 2), 800);
    REQUIRE_EQUALS(c1(1, 0), 800);
    REQUIRE_EQUALS(c1(1, 1), 800);
    REQUIRE_EQUALS(c1(1, 2), 800);
    REQUIRE_EQUALS(c1(2, 0), 800);
    REQUIRE_EQUALS(c1(2, 1), 800);
    REQUIRE_EQUALS(c1(2, 2), 800);
}

#endif
