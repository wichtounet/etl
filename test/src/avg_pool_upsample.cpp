//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(4, 4, etl::values<Z>(11, 6, 10, 8, 9, 16, 14, 1, 2, 7, 12, 13, 3, 15, 5, 4));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    auto output = etl::make_dyn_matrix<Z>(2, 2);
    output = etl::avg_pool_2d(input, 2, 2);

    auto result = etl::make_dyn_matrix<Z>(4, 4);
    result = etl::avg_pool_upsample_2d(input, output, errors, 2, 2);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 25);
    REQUIRE_EQUALS(result(0, 2), 50);
    REQUIRE_EQUALS(result(0, 3), 50);

    REQUIRE_EQUALS(result(1, 0), 25);
    REQUIRE_EQUALS(result(1, 1), 25);
    REQUIRE_EQUALS(result(1, 2), 50);
    REQUIRE_EQUALS(result(1, 3), 50);

    REQUIRE_EQUALS(result(2, 0), 75);
    REQUIRE_EQUALS(result(2, 1), 75);
    REQUIRE_EQUALS(result(2, 2), 100);
    REQUIRE_EQUALS(result(2, 3), 100);

    REQUIRE_EQUALS(result(3, 0), 75);
    REQUIRE_EQUALS(result(3, 1), 75);
    REQUIRE_EQUALS(result(3, 2), 100);
    REQUIRE_EQUALS(result(3, 3), 100);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(6, 6, etl::values(11, 32, 28, 33, 3, 26, 21, 35, 29, 25, 4, 5, 20, 34, 6, 18, 17, 
                                                  7, 10, 12, 9, 1, 8, 13, 2, 19, 16, 14, 23, 30, 22, 24, 15, 278, 36, 31));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(2, 2);
    output = etl::avg_pool_2d(input, 3, 3);

    etl::dyn_matrix<Z, 2> result(6, 6);
    result = etl::avg_pool_upsample_2d(input, output, errors, 3, 3);

    REQUIRE_EQUALS_APPROX(result(0, 0), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(0, 1), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(0, 2), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(0, 3), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(0, 4), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(0, 5), Z(200.0) / 9);

    REQUIRE_EQUALS_APPROX(result(1, 0), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(1, 1), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(1, 2), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(1, 3), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(1, 4), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(1, 5), Z(200.0) / 9);

    REQUIRE_EQUALS_APPROX(result(2, 0), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(2, 1), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(2, 2), Z(100.0) / 9);
    REQUIRE_EQUALS_APPROX(result(2, 3), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(2, 4), Z(200.0) / 9);
    REQUIRE_EQUALS_APPROX(result(2, 5), Z(200.0) / 9);

    REQUIRE_EQUALS_APPROX(result(3, 0), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(3, 1), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(3, 2), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(3, 3), Z(400.0) / 9);
    REQUIRE_EQUALS_APPROX(result(3, 4), Z(400.0) / 9);
    REQUIRE_EQUALS_APPROX(result(3, 5), Z(400.0) / 9);

    REQUIRE_EQUALS_APPROX(result(4, 0), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(4, 1), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(4, 2), Z(300.0) / 9);
    REQUIRE_EQUALS_APPROX(result(4, 3), Z(400.0) / 9);
    REQUIRE_EQUALS_APPROX(result(4, 4), Z(400.0) / 9);
    REQUIRE_EQUALS_APPROX(result(4, 5), Z(400.0) / 9);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(6, 4, etl::values<Z>(17, 16, 24, 8, 21, 19, 3, 10, 11, 9, 12, 7, 6, 23, 5, 15, 20, 18, 4, 2, 1, 13, 14, 22));
    etl::dyn_matrix<Z, 2> errors(1, 4, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(1, 4);
    output = etl::avg_pool_2d(input, 6, 1);

    etl::dyn_matrix<Z, 2> result(6, 4);
    result = etl::avg_pool_upsample_2d(input, output, errors, 6, 1);

    REQUIRE_EQUALS_APPROX(result(0, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(0, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(0, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(0, 3), Z(200.0) / 3);

    REQUIRE_EQUALS_APPROX(result(1, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(1, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(1, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(1, 3), Z(200.0) / 3);

    REQUIRE_EQUALS_APPROX(result(2, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(2, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(2, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(2, 3), Z(200.0) / 3);

    REQUIRE_EQUALS_APPROX(result(3, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(3, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(3, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(3, 3), Z(200.0) / 3);

    REQUIRE_EQUALS_APPROX(result(4, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(4, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(4, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(4, 3), Z(200.0) / 3);

    REQUIRE_EQUALS_APPROX(result(4, 0), Z(100.0) / 6);
    REQUIRE_EQUALS_APPROX(result(4, 1), Z(100.0) / 3);
    REQUIRE_EQUALS_APPROX(result(4, 2), Z(200.0) / 4);
    REQUIRE_EQUALS_APPROX(result(4, 3), Z(200.0) / 3);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(3, 3, etl::values<Z>(7, 8, 2, 3, 5, 1, 4, 6, 9));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(2, 2);
    output = etl::avg_pool_2d(input, 2, 2, 1, 1);

    etl::dyn_matrix<Z, 2> result(3, 3);
    result = etl::avg_pool_upsample_2d(input, output, errors, 2, 2, 1, 1);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 75);
    REQUIRE_EQUALS(result(0, 2), 50);

    REQUIRE_EQUALS(result(1, 0), 100);
    REQUIRE_EQUALS(result(1, 1), 250);
    REQUIRE_EQUALS(result(1, 2), 150);

    REQUIRE_EQUALS(result(2, 0), 75);
    REQUIRE_EQUALS(result(2, 1), 175);
    REQUIRE_EQUALS(result(2, 2), 100);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(2, 2, etl::values<Z>(2, 3, 4, 1));
    etl::dyn_matrix<Z, 2> errors(2, 2, etl::values<Z>(100, 200, 300, 400));

    etl::dyn_matrix<Z, 2> output(2, 2);
    output = etl::avg_pool_2d(input, 2, 2, 2, 2, 1, 1);

    etl::dyn_matrix<Z, 2> result(2, 2);
    result = etl::avg_pool_upsample_2d(input, output, errors, 2, 2, 2, 2, 1, 1);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 50);

    REQUIRE_EQUALS(result(1, 0), 75);
    REQUIRE_EQUALS(result(1, 1), 100);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/5", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(2, 2, etl::values<Z>(1, 2, 3, 4));
    etl::dyn_matrix<Z, 2> errors(3, 3, etl::values<Z>(100, 200, 300, 400, 500, 600, 700, 800, 900));

    etl::dyn_matrix<Z, 2> output(3, 3);
    output = etl::avg_pool_2d(input, 2, 2, 1, 1, 1, 1);

    etl::dyn_matrix<Z, 2> result(2, 2);
    result = etl::avg_pool_upsample_2d(input, output, errors, 2, 2, 1, 1, 1, 1);

    REQUIRE_EQUALS(result(0, 0), 300);
    REQUIRE_EQUALS(result(0, 1), 400);

    REQUIRE_EQUALS(result(1, 0), 600);
    REQUIRE_EQUALS(result(1, 1), 700);
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/6", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> input(2, 2, etl::values<Z>(1, 2, 3, 4));
    etl::dyn_matrix<Z, 2> errors(5, 5, etl::values<Z>(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500));

    etl::dyn_matrix<Z, 2> output(5, 5);
    output = etl::avg_pool_2d(input, 2, 2, 1, 1, 2, 2);

    etl::dyn_matrix<Z, 2> result(2, 2);
    result = etl::avg_pool_upsample_2d(input, output, errors, 2, 2, 1, 1, 2, 2);

    REQUIRE_EQUALS(result(0, 0), 1000);
    REQUIRE_EQUALS(result(0, 1), 1100);

    REQUIRE_EQUALS(result(1, 0), 1500);
    REQUIRE_EQUALS(result(1, 1), 1600);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> input({14, 1, 4, 15, 2, 6, 3, 11, 16, 10, 13, 7, 5, 12, 9, 8});
    etl::fast_matrix<Z, 2, 2> errors({100, 200, 300, 400});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::avg_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::avg_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 25);
    REQUIRE_EQUALS(result(0, 2), 50);
    REQUIRE_EQUALS(result(0, 3), 50);

    REQUIRE_EQUALS(result(1, 0), 25);
    REQUIRE_EQUALS(result(1, 1), 25);
    REQUIRE_EQUALS(result(1, 2), 50);
    REQUIRE_EQUALS(result(1, 3), 50);

    REQUIRE_EQUALS(result(2, 0), 75);
    REQUIRE_EQUALS(result(2, 1), 75);
    REQUIRE_EQUALS(result(2, 2), 100);
    REQUIRE_EQUALS(result(2, 3), 100);

    REQUIRE_EQUALS(result(3, 0), 75);
    REQUIRE_EQUALS(result(3, 1), 75);
    REQUIRE_EQUALS(result(3, 2), 100);
    REQUIRE_EQUALS(result(3, 3), 100);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 6, 2> input({1, 6, 5, 11, 9, 2, 7, 10, 8, 12, 3, 4});
    etl::fast_matrix<Z, 3, 1> errors({100, 200, 300});

    etl::fast_matrix<Z, 3, 1> output;
    output = etl::avg_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 6, 2> result;
    result = etl::avg_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 25);

    REQUIRE_EQUALS(result(1, 0), 25);
    REQUIRE_EQUALS(result(1, 1), 25);

    REQUIRE_EQUALS(result(2, 0), 50);
    REQUIRE_EQUALS(result(2, 1), 50);

    REQUIRE_EQUALS(result(3, 0), 50);
    REQUIRE_EQUALS(result(3, 1), 50);

    REQUIRE_EQUALS(result(4, 0), 75);
    REQUIRE_EQUALS(result(4, 1), 75);

    REQUIRE_EQUALS(result(5, 0), 75);
    REQUIRE_EQUALS(result(5, 1), 75);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 8, 2> input({1, 8, 5, 13, 16, 11, 9, 10, 15, 12, 7, 2, 6, 4, 14, 3});
    etl::fast_matrix<Z, 4, 2> errors({100, 200, 300, 400, 500, 600, 700, 800});

    etl::fast_matrix<Z, 4, 2> output;
    output = etl::avg_pool_2d<2, 1>(input);

    etl::fast_matrix<Z, 8, 2> result;
    result = etl::avg_pool_upsample_2d<2, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 50);
    REQUIRE_EQUALS(result(0, 1), 100);

    REQUIRE_EQUALS(result(1, 0), 50);
    REQUIRE_EQUALS(result(1, 1), 100);

    REQUIRE_EQUALS(result(2, 0), 150);
    REQUIRE_EQUALS(result(2, 1), 200);

    REQUIRE_EQUALS(result(3, 0), 150);
    REQUIRE_EQUALS(result(3, 1), 200);

    REQUIRE_EQUALS(result(4, 0), 250);
    REQUIRE_EQUALS(result(4, 1), 300);

    REQUIRE_EQUALS(result(5, 0), 250);
    REQUIRE_EQUALS(result(5, 1), 300);

    REQUIRE_EQUALS(result(6, 0), 350);
    REQUIRE_EQUALS(result(6, 1), 400);

    REQUIRE_EQUALS(result(7, 0), 350);
    REQUIRE_EQUALS(result(7, 1), 400);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> input{5, 16, 4, 6, 2, 13, 10, 7, 11, 8, 9, 15, 3, 14, 1, 12};
    etl::fast_matrix<Z, 3, 3> errors({100, 200, 300, 400, 500, 600, 700, 800, 900});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::avg_pool_2d<2, 2, 2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::avg_pool_upsample_2d<2, 2, 2, 2, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 25);
    REQUIRE_EQUALS(result(0, 1), 50);
    REQUIRE_EQUALS(result(0, 2), 50);
    REQUIRE_EQUALS(result(0, 3), 75);

    REQUIRE_EQUALS(result(1, 0), 100);
    REQUIRE_EQUALS(result(1, 1), 125);
    REQUIRE_EQUALS(result(1, 2), 125);
    REQUIRE_EQUALS(result(1, 3), 150);

    REQUIRE_EQUALS(result(2, 0), 100);
    REQUIRE_EQUALS(result(2, 1), 125);
    REQUIRE_EQUALS(result(2, 2), 125);
    REQUIRE_EQUALS(result(2, 3), 150);

    REQUIRE_EQUALS(result(3, 0), 175);
    REQUIRE_EQUALS(result(3, 1), 200);
    REQUIRE_EQUALS(result(3, 2), 200);
    REQUIRE_EQUALS(result(3, 3), 225);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> input{1, 2, 3, 4};
    etl::fast_matrix<Z, 3, 3> errors({100, 200, 300, 400, 500, 600, 700, 800, 900});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::avg_pool_2d<2, 2, 1, 1, 1, 1>(input);

    etl::fast_matrix<Z, 2, 2> result;
    result = etl::avg_pool_upsample_2d<2, 2, 1, 1, 1, 1>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 300);
    REQUIRE_EQUALS(result(0, 1), 400);

    REQUIRE_EQUALS(result(1, 0), 600);
    REQUIRE_EQUALS(result(1, 1), 700);
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> input{1, 2, 3, 4};
    etl::fast_matrix<Z, 5, 5> errors({100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500});

    etl::fast_matrix<Z, 5, 5> output;
    output = etl::avg_pool_2d<2, 2, 1, 1, 2, 2>(input);

    etl::fast_matrix<Z, 2, 2> result;
    result = etl::avg_pool_upsample_2d<2, 2, 1, 1, 2, 2>(input, output, errors);

    REQUIRE_EQUALS(result(0, 0), 1000);
    REQUIRE_EQUALS(result(0, 1), 1100);

    REQUIRE_EQUALS(result(1, 0), 1500);
    REQUIRE_EQUALS(result(1, 1), 1600);
}
