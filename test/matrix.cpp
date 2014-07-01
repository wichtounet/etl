#include "catch.hpp"

#include "etl/fast_matrix.hpp"

//{{{ Init tests

TEST_CASE( "fast_matrix/init_1", "fast_matrix::fast_matrix(T)" ) {
    etl::fast_matrix<double, 2, 2> test_matrix(3.3);

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "fast_matrix/init_2", "fast_matrix::operator=(T)" ) {
    etl::fast_matrix<double, 2, 2> test_matrix;

    test_matrix = 3.3;

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "fast_matrix/init_3", "fast_matrix::fast_matrix(initializer_list)" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {1.0, 3.0, 5.0, 2.0};

    REQUIRE(test_matrix.size() == 4);

    REQUIRE(test_matrix[0] == 1.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 5.0);
}

//}}} Init tests

//{{{ Binary operators test

TEST_CASE( "fast_matrix/add_scalar_1", "fast_matrix::operator+" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 + test_matrix;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "fast_matrix/add_scalar_2", "fast_matrix::operator+" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix + 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "fast_matrix/add", "fast_matrix::operator+" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<double, 2, 2> c = a + b;

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "fast_matrix/sub_scalar_1", "fast_matrix::operator+" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 - test_matrix;

    REQUIRE(test_matrix[0] == 2.0);
    REQUIRE(test_matrix[1] == -1.0);
    REQUIRE(test_matrix[2] == -4.5);
}

TEST_CASE( "fast_matrix/sub_scalar_2", "fast_matrix::operator+" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix - 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEST_CASE( "fast_matrix/sub", "fast_matrix::operator-" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<double, 2, 2> c = a - b;

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "fast_matrix/mul_scalar_1", "fast_matrix::operator*" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 * test_matrix;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEST_CASE( "fast_matrix/mul_scalar_2", "fast_matrix::operator*" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix * 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEST_CASE( "fast_matrix/mul", "fast_matrix::operator*" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<double, 2, 2> c = a * b;

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "fast_matrix/div_scalar_1", "fast_matrix::operator/" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix / 2.5;

    REQUIRE(test_matrix[0] == -1.0 / 2.5);
    REQUIRE(test_matrix[1] ==  2.0 / 2.5);
    REQUIRE(test_matrix[2] ==  5.0 / 2.5);
}

TEST_CASE( "fast_matrix/div_scalar_2", "fast_matrix::operator/" ) {
    etl::fast_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 / test_matrix;

    REQUIRE(test_matrix[0] == 2.5 / -1.0);
    REQUIRE(test_matrix[1] == 2.5 /  2.0);
    REQUIRE(test_matrix[2] == 2.5 /  5.0);
}

TEST_CASE( "fast_matrix/div", "fast_matrix::operator/" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<double, 2, 2> c = a / b;

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "fast_matrix/mod_scalar_1", "fast_matrix::operator%" ) {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = test_matrix % 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "fast_matrix/mod_scalar_2", "fast_matrix::operator%" ) {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = 2 % test_matrix;

    REQUIRE(test_matrix[0] == 2 % -1);
    REQUIRE(test_matrix[1] == 2 %  2);
    REQUIRE(test_matrix[2] == 2 %  5);
}

TEST_CASE( "fast_matrix/mod", "fast_matrix::operator*" ) {
    etl::fast_matrix<int, 2, 2> a = {-1, 2, 5, 1};
    etl::fast_matrix<int, 2, 2> b = {2, 3, 4, 1};

    etl::fast_matrix<int, 2, 2> c = a % b;

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "fast_matrix/log", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<double, 2, 2> d = log(a);

    REQUIRE(d[0] == log(-1.0));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "fast_matrix/abs", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<double, 2, 2> d = abs(a);

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_matrix/sign", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<double, 2, 2> d = sign(a);

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_matrix/unary_unary", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_matrix<double, 2, 2> d = abs(sign(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_matrix/unary_binary_1", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<double, 2, 2> d = abs(a + a);

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_matrix/unary_binary_2", "fast_matrix::abs" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<double, 2, 2> d = abs(a) + a;

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

//}}} Unary operators test

//{{{ Complex tests

TEST_CASE( "fast_matrix/complex", "fast_matrix::complex" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_matrix<double, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_matrix<double, 2, 2> d = 2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c);

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEST_CASE( "fast_matrix/complex_2", "fast_matrix::complex" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_matrix<double, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_matrix<double, 2, 2> d = 2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c);

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(5.8273));
}

TEST_CASE( "fast_matrix/complex_3", "fast_matrix::complex" ) {
    etl::fast_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_matrix<double, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_matrix<double, 2, 2> d = 2.5 / (a * b);

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests