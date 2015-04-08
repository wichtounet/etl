//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

//{{{ Init tests

TEMPLATE_TEST_CASE_2( "dyn_vector/init_1", "dyn_vector::dyn_vector(T)", Z, double, float) {
    etl::dyn_vector<Z> test_vector(4, 3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == Z(3.3));
        REQUIRE(test_vector(i) == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "dyn_vector/init_2", "dyn_vector::operator=(T)", Z, double, float) {
    etl::dyn_vector<Z> test_vector(4);

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == Z(3.3));
        REQUIRE(test_vector(i) == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "dyn_vector/init_3", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    etl::dyn_vector<Z> test_vector({1.0, 2.0, 3.0});

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

//}}} Init tests


//{{{ Binary operators test

TEMPLATE_TEST_CASE_2( "dyn_vector/add_scalar_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/add_scalar_2", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/add_scalar_3", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/add_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/add_2", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sub_scalar_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE(test_vector[0] == 2.0);
    REQUIRE(test_vector[1] == -1.0);
    REQUIRE(test_vector[2] == -4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sub_scalar_2", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sub_scalar_3", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sub_1", "dyn_vector::operator-", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sub_2", "dyn_vector::operator-=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mul_scalar_1", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mul_scalar_2", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mul_scalar_3", "dyn_vector::operator*=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mul_1", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c(a >> b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mul_2", "dyn_vector::operator*=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/div_scalar_1", "dyn_vector::operator/", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / Z(2.5);

    REQUIRE(test_vector[0] == Z(-1.0 / 2.5));
    REQUIRE(test_vector[1] == Z(2.0 / 2.5));
    REQUIRE(test_vector[2] == Z(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/div_scalar_2", "dyn_vector::operator/", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE(test_vector[0] == 2.5 / -1.0);
    REQUIRE(test_vector[1] == 2.5 /  2.0);
    REQUIRE(test_vector[2] == 2.5 /  5.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/div_scalar_3", "dyn_vector::operator/=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= Z(2.5);

    REQUIRE(test_vector[0] == Z(-1.0 / 2.5));
    REQUIRE(test_vector[1] == Z(2.0 / 2.5));
    REQUIRE(test_vector[2] == Z(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/div_1", "dyn_vector::operator/", Z, double, float){
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c(a / b);

    REQUIRE(c[0] == Z(-1.0 / 2.5));
    REQUIRE(c[1] == Z(2.0 / 3.0));
    REQUIRE(c[2] == Z(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/div_2", "dyn_vector::operator/=", Z, double, float){
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE(a[0] == Z(-1.0 / 2.5));
    REQUIRE(a[1] == Z(2.0 / 3.0));
    REQUIRE(a[2] == Z(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mod_scalar_1", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mod_scalar_2", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE(test_vector[0] == 2 % -1);
    REQUIRE(test_vector[1] == 2 %  2);
    REQUIRE(test_vector[2] == 2 %  5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mod_scalar_3", "dyn_vector::operator%=", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mod_1", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    etl::dyn_vector<int> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/mod_2", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEMPLATE_TEST_CASE_2( "dyn_vector/log", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};

    etl::dyn_vector<Z> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == std::log(Z(2.0)));
    REQUIRE(d[2] == std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/abs", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sign", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/unary_unary", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/unary_binary_1", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/unary_binary_2", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sigmoid", "dyn_vector::sigmoid", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(Z(-1.0)));
    REQUIRE(d[1] == etl::logistic_sigmoid(Z(2.0)));
    REQUIRE(d[2] == etl::logistic_sigmoid(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/softplus", "dyn_vector::softplus", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::softplus(a));

    REQUIRE(d[0] == etl::softplus(Z(-1.0)));
    REQUIRE(d[1] == etl::softplus(Z(2.0)));
    REQUIRE(d[2] == etl::softplus(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/exp", "dyn_vector::exp", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::exp(a));

    REQUIRE(d[0] == std::exp(Z(-1.0)));
    REQUIRE(d[1] == std::exp(Z(2.0)));
    REQUIRE(d[2] == std::exp(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/max", "dyn_vector::max", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::max(a,1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/min", "dyn_vector::min", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::min(a,1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2( "dyn_vector/bernoulli", "dyn_vector::bernoulli", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
}


//}}} Unary operators test

//{{{ Reductions

TEMPLATE_TEST_CASE_2( "dyn_vector/sum", "sum", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sum_2", "sum", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/sum_3", "sum", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/dot_1", "sum", Z, double, float) {
    etl::dyn_vector<Z> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<Z> b({2.0, 3.0, 2.0});

    auto d = dot(a, b);

    REQUIRE(d == 21.0);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/dot_2", "sum", Z, double, float) {
    etl::dyn_vector<Z> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<Z> b({2.0, 3.0, 2.0});

    auto d = dot(a, -1 * b);

    REQUIRE(d == -21.0);
}

//}}} Reductions

//{{{ Complex tests

TEMPLATE_TEST_CASE_2( "dyn_vector/complex", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<Z> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<Z> d(2.5 * ((a >> b) / (a + c)) / (1.5 * scale(a, b) / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/complex_2", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {1.1, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, -3.0, 4.0};
    etl::dyn_vector<Z> c = {2.2, 3.0, 3.5};

    etl::dyn_vector<Z> d(2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEMPLATE_TEST_CASE_2( "dyn_vector/complex_3", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<Z> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<Z> d(2.5 / (a >> b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests

//{{{ Complex content

TEMPLATE_TEST_CASE_2( "dyn_vector/complex_content_1", "dyn_vector<dyn_matrix>>", Z, double, float) {
    etl::dyn_vector<etl::dyn_matrix<Z>> a(11, etl::dyn_matrix<Z>(3,3));

    //It is enough for this test to compile
    REQUIRE(true);
}

TEMPLATE_TEST_CASE_2( "dyn_vector/complex_content_2", "vector<dyn_vector<dyn_matrix>>>", Z, double, float) {
    std::vector<etl::dyn_vector<etl::dyn_matrix<Z>>> a;

    a.emplace_back(11, etl::dyn_matrix<Z>(3,3));

    //It is enough for this test to compile
    REQUIRE(true);
}

//}}} Complex content

TEMPLATE_TEST_CASE_2( "dyn_vector/swap_1", "dyn_vector::swap", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    using std::swap;
    swap(a, b);

    REQUIRE(a[0] == 2.5);
    REQUIRE(a[1] == 3.0);
    REQUIRE(a[2] == 4.0);

    REQUIRE(b[0] == -1.0);
    REQUIRE(b[1] == 2.0);
    REQUIRE(b[2] == 5.0);
}
