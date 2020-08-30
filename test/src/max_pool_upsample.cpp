//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(4, 4, etl::values<Z>(14, 13, 7, 4, 9, 5, 10, 11, 16, 2, 8, 6, 15, 12, 1, 3));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    auto output = etl::make_dyn_matrix<Z>(2, 2);
    output = etl::max_pool_2d(input, 2, 2);

    auto result = etl::make_dyn_matrix<Z>(4, 4);
    result = etl::max_pool_upsample_2d(input, output, errors, 2, 2);

    REQUIRE_EQUALS(result(0, 0), 100);
    REQUIRE_EQUALS(result(0, 1), 0);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);
    REQUIRE_EQUALS(result(1, 3), 200);

    REQUIRE_EQUALS(result(2, 0), 300);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 400);
    REQUIRE_EQUALS(result(2, 3), 0);

    REQUIRE_EQUALS(result(3, 0), 0);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max2/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(6, 6, etl::values<Z>(14, 13, 17, 22, 23, 5,
                                                     18, 34, 16, 2, 36, 28,
                                                     15, 29, 1, 3, 30, 33,
                                                     7, 25, 9, 4, 26, 24,
                                                     31, 21, 12, 6, 35, 20,
                                                     19, 11, 10, 32, 27, 8));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(2, 2);
    output = etl::max_pool_2d(input, 3, 3);

    etl::dyn_matrix<Z, 2> result(6, 6);
    result = etl::max_pool_upsample_2d(input, output, errors, 3, 3);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 0);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);
    REQUIRE_EQUALS(result(0, 4), 0);
    REQUIRE_EQUALS(result(0, 5), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 100);
    REQUIRE_EQUALS(result(1, 2), 0);
    REQUIRE_EQUALS(result(1, 3), 0);
    REQUIRE_EQUALS(result(1, 4), 200);
    REQUIRE_EQUALS(result(1, 5), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 0);
    REQUIRE_EQUALS(result(2, 3), 0);
    REQUIRE_EQUALS(result(2, 4), 0);
    REQUIRE_EQUALS(result(2, 5), 0);

    REQUIRE_EQUALS(result(3, 0), 0);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);
    REQUIRE_EQUALS(result(3, 4), 0);
    REQUIRE_EQUALS(result(3, 5), 0);

    REQUIRE_EQUALS(result(4, 0), 300);
    REQUIRE_EQUALS(result(4, 1), 0);
    REQUIRE_EQUALS(result(4, 2), 0);
    REQUIRE_EQUALS(result(4, 3), 0);
    REQUIRE_EQUALS(result(4, 4), 400);
    REQUIRE_EQUALS(result(4, 5), 0);

    REQUIRE_EQUALS(result(5, 0), 0);
    REQUIRE_EQUALS(result(5, 1), 0);
    REQUIRE_EQUALS(result(5, 2), 0);
    REQUIRE_EQUALS(result(5, 3), 0);
    REQUIRE_EQUALS(result(5, 4), 0);
    REQUIRE_EQUALS(result(5, 5), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(6, 4, etl::values<Z>(21, 24, 2, 5, 17, 22, 14, 19, 20, 16, 10, 18, 23, 1, 4, 3, 6, 7, 13, 9, 12, 15, 8, 11));
    etl::dyn_matrix<Z, 2> errors(1, 4, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(1, 4);
    output = etl::max_pool_2d(input, 6, 1);

    etl::dyn_matrix<Z, 2> result(6, 4);
    result = etl::max_pool_upsample_2d(input, output, errors, 6, 1);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 200);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 300);
    REQUIRE_EQUALS(result(1, 3), 400);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 0);
    REQUIRE_EQUALS(result(2, 3), 0);

    REQUIRE_EQUALS(result(3, 0), 100);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);

    REQUIRE_EQUALS(result(4, 0), 0);
    REQUIRE_EQUALS(result(4, 1), 0);
    REQUIRE_EQUALS(result(4, 2), 0);
    REQUIRE_EQUALS(result(4, 3), 0);

    REQUIRE_EQUALS(result(5, 0), 0);
    REQUIRE_EQUALS(result(5, 1), 0);
    REQUIRE_EQUALS(result(5, 2), 0);
    REQUIRE_EQUALS(result(5, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(3, 3, etl::values<Z>(9, 8, 5, 7, 6, 4, 2, 3, 1));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(2, 2);
    output = etl::max_pool_2d(input, 2, 2, 1, 1);

    etl::dyn_matrix<Z, 2> result(3, 3);
    result = etl::max_pool_upsample_2d(input, output, errors, 2, 2, 1, 1);

    REQUIRE_EQUALS(result(0, 0), 100);
    REQUIRE_EQUALS(result(0, 1), 200);
    REQUIRE_EQUALS(result(0, 2), 0);

    REQUIRE_EQUALS(result(1, 0), 300);
    REQUIRE_EQUALS(result(1, 1), 400);
    REQUIRE_EQUALS(result(1, 2), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max2/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(4, 4, etl::values<Z>(12, 11, 2, 5, 6, 15, 14, 13, 9, 16, 10, 7, 8, 1, 4, 3));
    etl::dyn_matrix<Z, 2> errors(4, 4, etl::values<Z>(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600));

    etl::dyn_matrix<Z, 2> output(4, 4);
    output = etl::max_pool_2d(input, 2, 2, 2, 2, 2, 2);

    etl::dyn_matrix<Z, 2> result(4, 4);
    result = etl::max_pool_upsample_2d(input, output, errors, 2, 2, 2, 2, 2, 2);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 0);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 600);
    REQUIRE_EQUALS(result(1, 2), 700);
    REQUIRE_EQUALS(result(1, 3), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 1000);
    REQUIRE_EQUALS(result(2, 2), 1100);
    REQUIRE_EQUALS(result(2, 3), 0);

    REQUIRE_EQUALS(result(3, 0), 0);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_derivative/max2/0", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> input({6.0, 7.0, 2.0, 9.0, 3.0, 5.0, 4.0, 1.0, 8.0});
    etl::fast_matrix<Z, 2, 2> errors({100.0, 200.0, 300.0, 400.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 3, 3> result;
    result = etl::max_pool_upsample_2d<2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 200);
    REQUIRE_EQUALS(result(0, 2), 0);

    REQUIRE_EQUALS(result(1, 0), 400);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 400);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> input({12, 11, 2, 5, 6, 15, 14, 13, 9, 16, 10, 7, 8, 1, 4, 3});
    etl::fast_matrix<Z, 2, 2> errors({100.0, 200.0, 300.0, 400.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 0);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 100);
    REQUIRE_EQUALS(result(1, 2), 200);
    REQUIRE_EQUALS(result(1, 3), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 300);
    REQUIRE_EQUALS(result(2, 2), 400);
    REQUIRE_EQUALS(result(2, 3), 0);

    REQUIRE_EQUALS(result(3, 0), 0);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 6, 4> input({21, 24, 2, 5, 17, 22, 14, 19, 20, 16, 10, 18, 23, 1, 4, 3, 6, 7, 13, 9, 12, 15, 8, 11});
    etl::fast_matrix<Z, 3, 2> errors({100.0, 200.0, 300.0, 400.0, 500.0, 600.0});

    etl::fast_matrix<Z, 3, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 6, 4> result;
    result = etl::max_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 100);
    REQUIRE_EQUALS(result(0, 2), 0);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);
    REQUIRE_EQUALS(result(1, 3), 200);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 0);
    REQUIRE_EQUALS(result(2, 3), 400);

    REQUIRE_EQUALS(result(3, 0), 300);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 0);
    REQUIRE_EQUALS(result(3, 3), 0);

    REQUIRE_EQUALS(result(4, 0), 0);
    REQUIRE_EQUALS(result(4, 1), 0);
    REQUIRE_EQUALS(result(4, 2), 600);
    REQUIRE_EQUALS(result(4, 3), 0);

    REQUIRE_EQUALS(result(5, 0), 0);
    REQUIRE_EQUALS(result(5, 1), 500);
    REQUIRE_EQUALS(result(5, 2), 0);
    REQUIRE_EQUALS(result(5, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 6, 4> input({2, 12, 19, 6, 15, 11, 13, 8, 23, 17, 4, 9, 20, 1, 21, 3, 18, 5, 22, 24, 7, 14, 16, 10});

    etl::fast_matrix<Z, 3, 4> errors({100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200});
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 3, 4> output;
    output = etl::max_pool_2d<2, 1>(input);

    etl::fast_matrix<Z, 6, 4> result;
    result = etl::max_pool_upsample_2d<2, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 200);
    REQUIRE_EQUALS(result(0, 2), 300);
    REQUIRE_EQUALS(result(0, 3), 0);

    REQUIRE_EQUALS(result(1, 0), 100);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);
    REQUIRE_EQUALS(result(1, 3), 400);

    REQUIRE_EQUALS(result(2, 0), 500);
    REQUIRE_EQUALS(result(2, 1), 600);
    REQUIRE_EQUALS(result(2, 2), 0);
    REQUIRE_EQUALS(result(2, 3), 800);

    REQUIRE_EQUALS(result(3, 0), 0);
    REQUIRE_EQUALS(result(3, 1), 0);
    REQUIRE_EQUALS(result(3, 2), 700);
    REQUIRE_EQUALS(result(3, 3), 0);

    REQUIRE_EQUALS(result(4, 0), 900);
    REQUIRE_EQUALS(result(4, 1), 0);
    REQUIRE_EQUALS(result(4, 2), 1100);
    REQUIRE_EQUALS(result(4, 3), 1200);

    REQUIRE_EQUALS(result(5, 0), 0);
    REQUIRE_EQUALS(result(5, 1), 1000);
    REQUIRE_EQUALS(result(5, 2), 0);
    REQUIRE_EQUALS(result(5, 3), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> input({2, 9, 7, 5, 4, 3, 7, 8, 1});
    etl::fast_matrix<Z, 2, 2> errors({100, 200, 300, 400});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 3, 3> result;
    result = etl::max_pool_upsample_2d<2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 300);
    REQUIRE_EQUALS(result(0, 2), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 700);
    REQUIRE_EQUALS(result(2, 2), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> input({3, 2, 4, 1});
    etl::fast_matrix<Z, 2, 2> errors({100, 200, 300, 400});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2, 2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 2, 2> result;
    result = etl::max_pool_upsample_2d<2, 2, 2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 100);
    REQUIRE_EQUALS(result(0, 1), 200);

    REQUIRE_EQUALS(result(1, 0), 300);
    REQUIRE_EQUALS(result(1, 1), 400);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> input({2, 8, 7, 5, 4, 3, 7, 8, 1});
    etl::fast_matrix<Z, 2, 2> errors({100, 200, 300, 400});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 3, 3> result;
    result = etl::max_pool_upsample_2d<2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 300);
    REQUIRE_EQUALS(result(0, 2), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 0);
    REQUIRE_EQUALS(result(1, 2), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 700);
    REQUIRE_EQUALS(result(2, 2), 0);
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/7", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> input({2, 8, 7, 5, 9, 3, 7, 8, 1});
    etl::fast_matrix<Z, 2, 2> errors({100, 200, 300, 400});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 3, 3> result;
    result = etl::max_pool_upsample_2d<2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 0);
    REQUIRE_EQUALS(result(0, 1), 0);
    REQUIRE_EQUALS(result(0, 2), 0);

    REQUIRE_EQUALS(result(1, 0), 0);
    REQUIRE_EQUALS(result(1, 1), 1000);
    REQUIRE_EQUALS(result(1, 2), 0);

    REQUIRE_EQUALS(result(2, 0), 0);
    REQUIRE_EQUALS(result(2, 1), 0);
    REQUIRE_EQUALS(result(2, 2), 0);
}
