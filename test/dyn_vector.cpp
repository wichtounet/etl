//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/dyn_vector.hpp"

//{{{ Init tests

TEST_CASE( "dyn_vector/init_1", "dyn_vector::dyn_vector(T)" ) {
    etl::dyn_vector<double> test_vector(4, 3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
        REQUIRE(test_vector(i) == 3.3);
    }
}

TEST_CASE( "dyn_vector/init_2", "dyn_vector::operator=(T)" ) {
    etl::dyn_vector<double> test_vector(4);

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
        REQUIRE(test_vector(i) == 3.3);
    }
}

TEST_CASE( "dyn_vector/init_3", "dyn_vector::dyn_vector(initializer_list)" ) {
    etl::dyn_vector<double> test_vector({1.0, 2.0, 3.0});

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

//}}} Init tests


//{{{ Binary operators test

TEST_CASE( "dyn_vector/add_scalar_1", "dyn_vector::operator+" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "dyn_vector/add_scalar_2", "dyn_vector::operator+" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "dyn_vector/add_scalar_3", "dyn_vector::operator+=" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "dyn_vector/add_1", "dyn_vector::operator+" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<double> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "dyn_vector/add_2", "dyn_vector::operator+=" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEST_CASE( "dyn_vector/sub_scalar_1", "dyn_vector::operator+" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE(test_vector[0] == 2.0);
    REQUIRE(test_vector[1] == -1.0);
    REQUIRE(test_vector[2] == -4.5);
}

TEST_CASE( "dyn_vector/sub_scalar_2", "dyn_vector::operator+" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "dyn_vector/sub_scalar_3", "dyn_vector::operator+=" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "dyn_vector/sub_1", "dyn_vector::operator-" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<double> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "dyn_vector/sub_2", "dyn_vector::operator-=" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEST_CASE( "dyn_vector/mul_scalar_1", "dyn_vector::operator*" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "dyn_vector/mul_scalar_2", "dyn_vector::operator*" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "dyn_vector/mul_scalar_3", "dyn_vector::operator*=" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "dyn_vector/mul_1", "dyn_vector::operator*" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<double> c(a * b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "dyn_vector/mul_2", "dyn_vector::operator*=" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEST_CASE( "dyn_vector/div_scalar_1", "dyn_vector::operator/" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] ==  2.0 / 2.5);
    REQUIRE(test_vector[2] ==  5.0 / 2.5);
}

TEST_CASE( "dyn_vector/div_scalar_2", "dyn_vector::operator/" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE(test_vector[0] == 2.5 / -1.0);
    REQUIRE(test_vector[1] == 2.5 /  2.0);
    REQUIRE(test_vector[2] == 2.5 /  5.0);
}

TEST_CASE( "dyn_vector/div_scalar_3", "dyn_vector::operator/=" ) {
    etl::dyn_vector<double> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] == 2.0 / 2.5);
    REQUIRE(test_vector[2] == 5.0 / 2.5);
}

TEST_CASE( "dyn_vector/div_1", "dyn_vector::operator/"){
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<double> c(a / b);

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "dyn_vector/div_2", "dyn_vector::operator/="){
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE(a[0] == -1.0 / 2.5);
    REQUIRE(a[1] == 2.0 / 3.0);
    REQUIRE(a[2] == 5.0 / 4.0);
}

TEST_CASE( "dyn_vector/mod_scalar_1", "dyn_vector::operator%" ) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "dyn_vector/mod_scalar_2", "dyn_vector::operator%" ) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE(test_vector[0] == 2 % -1);
    REQUIRE(test_vector[1] == 2 %  2);
    REQUIRE(test_vector[2] == 2 %  5);
}

TEST_CASE( "dyn_vector/mod_scalar_3", "dyn_vector::operator%=" ) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "dyn_vector/mod_1", "dyn_vector::operator%" ) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    etl::dyn_vector<int> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "dyn_vector/mod_2", "dyn_vector::operator%" ) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "dyn_vector/log", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};

    etl::dyn_vector<double> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "dyn_vector/abs", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_vector/sign", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_vector/unary_unary", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_vector/unary_binary_1", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_vector/unary_binary_2", "dyn_vector::abs" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_vector/sigmoid", "dyn_vector::sigmoid" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(-1.0));
    REQUIRE(d[1] == etl::logistic_sigmoid(2.0));
    REQUIRE(d[2] == etl::logistic_sigmoid(0.0));
}

TEST_CASE( "dyn_vector/softplus", "dyn_vector::softplus" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::softplus(a));

    REQUIRE(d[0] == etl::softplus(-1.0));
    REQUIRE(d[1] == etl::softplus(2.0));
    REQUIRE(d[2] == etl::softplus(0.0));
}

TEST_CASE( "dyn_vector/exp", "dyn_vector::exp" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::exp(a));

    REQUIRE(d[0] == std::exp(-1.0));
    REQUIRE(d[1] == std::exp(2.0));
    REQUIRE(d[2] == std::exp(0.0));
}

TEST_CASE( "dyn_vector/max", "dyn_vector::max" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::max(a,1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
}

TEST_CASE( "dyn_vector/min", "dyn_vector::min" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::min(a,1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEST_CASE( "dyn_vector/bernoulli", "dyn_vector::bernoulli" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<double> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
}


//}}} Unary operators test

//{{{ Reductions

TEST_CASE( "dyn_vector/sum", "sum" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEST_CASE( "dyn_vector/sum_2", "sum" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEST_CASE( "dyn_vector/sum_3", "sum" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEST_CASE( "dyn_vector/dot_1", "sum" ) {
    etl::dyn_vector<double> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<double> b({2.0, 3.0, 2.0});

    auto d = dot(a, b);

    REQUIRE(d == 21.0);
}

TEST_CASE( "dyn_vector/dot_2", "sum" ) {
    etl::dyn_vector<double> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<double> b({2.0, 3.0, 2.0});

    auto d = dot(a, -1 * b);

    REQUIRE(d == -21.0);
}

//}}} Reductions

//{{{ Complex tests

TEST_CASE( "dyn_vector/complex", "dyn_vector::complex" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<double> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<double> d(2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEST_CASE( "dyn_vector/complex_2", "dyn_vector::complex" ) {
    etl::dyn_vector<double> a = {1.1, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, -3.0, 4.0};
    etl::dyn_vector<double> c = {2.2, 3.0, 3.5};

    etl::dyn_vector<double> d(2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEST_CASE( "dyn_vector/complex_3", "dyn_vector::complex" ) {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<double> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<double> d(2.5 / (a * b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests

//{{{ Complex content

TEST_CASE( "dyn_vector/complex_content_1", "dyn_vector<dyn_matrix>>" ) {
    etl::dyn_vector<etl::dyn_matrix<double>> a(11, etl::dyn_matrix<double>(3,3));

    //It is enough for this test to compile
    REQUIRE(true);
}

TEST_CASE( "dyn_vector/complex_content_2", "vector<dyn_vector<dyn_matrix>>>" ) {
    std::vector<etl::dyn_vector<etl::dyn_matrix<double>>> a;

    a.emplace_back(11, etl::dyn_matrix<double>(3,3));

    //It is enough for this test to compile
    REQUIRE(true);
}

//}}} Complex content