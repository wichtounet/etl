#include "catch.hpp"

#include "etl/fast_vector.hpp"

//{{{ Init tests

TEST_CASE( "fast_vector/init_1", "fast_vector::fast_vector(T)" ) {
    etl::fast_vector<double, 4> test_vector(3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
    }
}

TEST_CASE( "fast_vector/init_2", "fast_vector::operator=(T)" ) {
    etl::fast_vector<double, 4> test_vector;

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
    }
}

TEST_CASE( "fast_vector/init_3", "fast_vector::fast_vector(initializer_list)" ) {
    etl::fast_vector<double, 3> test_vector = {1.0, 2.0, 3.0};

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

//}}} Init tests

//{{{ Binary operators test

TEST_CASE( "fast_vector/add_scalar_1", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add_scalar_2", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a + b;

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "fast_vector/sub_scalar_1", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE(test_vector[0] == 2.0);
    REQUIRE(test_vector[1] == -1.0);
    REQUIRE(test_vector[2] == -4.5);
}

TEST_CASE( "fast_vector/sub_scalar_2", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "fast_vector/sub", "fast_vector::operator-" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a - b;

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "fast_vector/mul_scalar_1", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul_scalar_2", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a * b;

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "fast_vector/div_scalar_1", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] ==  2.0 / 2.5);
    REQUIRE(test_vector[2] ==  5.0 / 2.5);
}

TEST_CASE( "fast_vector/div_scalar_2", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE(test_vector[0] == 2.5 / -1.0);
    REQUIRE(test_vector[1] == 2.5 /  2.0);
    REQUIRE(test_vector[2] == 2.5 /  5.0);
}

TEST_CASE( "fast_vector/div", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a / b;

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "fast_vector/mod_scalar_1", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod_scalar_2", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE(test_vector[0] == 2 % -1);
    REQUIRE(test_vector[1] == 2 %  2);
    REQUIRE(test_vector[2] == 2 %  5);
}

TEST_CASE( "fast_vector/mod", "fast_vector::operator*" ) {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    etl::fast_vector<int, 3> c = a % b;

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "fast_vector/log", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};

    etl::fast_vector<double, 3> d = log(a);

    REQUIRE(d[0] == log(-1.0));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "fast_vector/abs", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d = abs(a);

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/sign", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d = sign(a);

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_unary", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d = abs(sign(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_binary_1", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d = abs(a + a);

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_binary_2", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d = abs(a) + a;

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

//}}} Unary operators test

//{{{ Reductions

TEST_CASE( "fast_vector/sum", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEST_CASE( "fast_vector/sum_2", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEST_CASE( "fast_vector/sum_3", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE(d == 19);
}

//}}} Reductions

//{{{ Complex tests

TEST_CASE( "fast_vector/complex", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<double, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<double, 3> d = 2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c);

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEST_CASE( "fast_vector/complex_2", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<double, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<double, 3> d = 2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c);

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(5.8273));
}

TEST_CASE( "fast_vector/complex_3", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<double, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<double, 3> d = 2.5 / (a * b);

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests