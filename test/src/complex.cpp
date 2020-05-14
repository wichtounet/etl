//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include "mmul_test.hpp"

#define CZ(a, b) std::complex<Z>(a, b)
#define ECZ(a, b) etl::complex<Z>(a, b)

ETL_TEST_CASE("etl_complex/1", "[complex]") {
    REQUIRE_EQUALS(sizeof(etl::complex<float>), sizeof(std::complex<float>));
    REQUIRE_EQUALS(sizeof(etl::complex<double>), sizeof(std::complex<double>));

    // This is done to ensure binary compatibility with std::complex

    etl::complex<float> a(3.3, 5.5);

    REQUIRE_EQUALS(reinterpret_cast<float(&)[2]>(a)[0], float(3.3));
    REQUIRE_EQUALS(reinterpret_cast<float(&)[2]>(a)[1], float(5.5));

    etl::complex<double> b(-2.3, 4.1);

    REQUIRE_EQUALS(reinterpret_cast<double(&)[2]>(b)[0], double(-2.3));
    REQUIRE_EQUALS(reinterpret_cast<double(&)[2]>(b)[1], double(4.1));
}

ETL_TEST_CASE("etl_complex/2", "[complex]") {
    etl::complex<float> e_a(3.3, 5.5);
    std::complex<float> s_a(3.3, 5.5);

    etl::complex<double> e_b(-2.3, 4.1);
    std::complex<double> s_b(-2.3, 4.1);

    REQUIRE_EQUALS_APPROX(std::abs(s_a), etl::abs(e_a));
    REQUIRE_EQUALS_APPROX(std::abs(s_b), etl::abs(e_b));

    REQUIRE_EQUALS_APPROX(std::sqrt(s_a).imag(), etl::sqrt(e_a).imag);
    REQUIRE_EQUALS_APPROX(std::sqrt(s_a).real(), etl::sqrt(e_a).real);
    REQUIRE_EQUALS_APPROX(std::sqrt(s_b).imag(), etl::sqrt(e_b).imag);
    REQUIRE_EQUALS_APPROX(std::sqrt(s_b).real(), etl::sqrt(e_b).real);
}

TEMPLATE_TEST_CASE_2("complex/std/1", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {-1.0, 2.0, 5.0};

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(-1.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a[1].real(), Z(2.0));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), Z(5.0));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(0.0));

    a[0] = 33.0;

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(33.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(0.0));

    a[0].imag(12.0);

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(33.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(12.0));

    a[0] = std::complex<Z>(1.0, 2.0);

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(1.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(2.0));

    a = std::complex<Z>(3.0, -2.0);

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(3.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(-2.0));
    REQUIRE_EQUALS_APPROX(a[1].real(), Z(3.0));
    REQUIRE_EQUALS_APPROX(a[1].imag(), Z(-2.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), Z(3.0));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(-2.0));
}

TEMPLATE_TEST_CASE_2("complex/std/2", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_vector<std::complex<Z>, 3> c;

    c = a >> b;

    REQUIRE_EQUALS(c[0], a[0] * b[0]);
    REQUIRE_EQUALS(c[1], a[1] * b[1]);
    REQUIRE_EQUALS(c[2], a[2] * b[2]);
}

TEMPLATE_TEST_CASE_2("complex/std/3", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_vector<std::complex<Z>, 3> c;

    c = a + b;

    REQUIRE_EQUALS_APPROX(c[0].real(), Z(1.33));
    REQUIRE_EQUALS_APPROX(c[0].imag(), Z(2.66));
    REQUIRE_EQUALS_APPROX(c[1].real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(c[1].imag(), Z(-2.0));
    REQUIRE_EQUALS_APPROX(c[2].real(), Z(0.5));
    REQUIRE_EQUALS_APPROX(c[2].imag(), Z(1.25));
}

TEMPLATE_TEST_CASE_2("complex/std/4", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_vector<std::complex<Z>, 3> c;

    c = a - b;

    REQUIRE_EQUALS_APPROX(c[0].real(), Z(0.67));
    REQUIRE_EQUALS_APPROX(c[0].imag(), Z(1.34));
    REQUIRE_EQUALS_APPROX(c[1].real(), Z(0.5));
    REQUIRE_EQUALS_APPROX(c[1].imag(), Z(-2.0));
    REQUIRE_EQUALS_APPROX(c[2].real(), Z(-0.5));
    REQUIRE_EQUALS_APPROX(c[2].imag(), Z(-0.25));
}

TEMPLATE_TEST_CASE_2("complex/std/5", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_vector<std::complex<Z>, 3> c;

    c = a >> b;

    REQUIRE_EQUALS_APPROX(c[0].real(), Z(-0.99));
    REQUIRE_EQUALS_APPROX(c[0].imag(), Z(1.32));
    REQUIRE_EQUALS_APPROX(c[1].real(), Z(1.5));
    REQUIRE_EQUALS_APPROX(c[1].imag(), Z(3.0));
    REQUIRE_EQUALS_APPROX(c[2].real(), Z(-0.375));
    REQUIRE_EQUALS_APPROX(c[2].imag(), Z(0.25));
}

TEMPLATE_TEST_CASE_2("complex/std/6", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_vector<std::complex<Z>, 3> c;

    c = a / b;

    REQUIRE_EQUALS_APPROX(c[0].real(), Z(3.0303030));
    REQUIRE_EQUALS_APPROX(c[0].imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(c[1].real(), Z(0.6666666));
    REQUIRE_EQUALS_APPROX(c[1].imag(), Z(1.3333333));
    REQUIRE_EQUALS_APPROX(c[2].real(), Z(0.461538));
    REQUIRE_EQUALS_APPROX(c[2].imag(), Z(0.3076923));
}

TEMPLATE_TEST_CASE_2("complex/std/7", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 1024> a;
    etl::fast_vector<std::complex<Z>, 1024> b;
    etl::fast_vector<std::complex<Z>, 1024> c;

    for (size_t i = 0; i < 1024; ++i) {
        a[i] = std::complex<Z>(i * 1099.66, (i - 32.3) * -23.04);
        b[i] = std::complex<Z>((i - 100) * 99.66, (i + 14.3) * 23.04);
    }

    c = a >> b;

    for (size_t i = 0; i < 1024; ++i) {
        REQUIRE_EQUALS_APPROX(c[i].real(), (a[i] * b[i]).real());
        REQUIRE_EQUALS_APPROX(c[i].imag(), (a[i] * b[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("complex/std/8", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 1024> a;
    etl::fast_vector<std::complex<Z>, 1024> b;
    etl::fast_vector<std::complex<Z>, 1024> c;

    for (size_t i = 0; i < 1024; ++i) {
        a[i] = std::complex<Z>(i * 1039.66, (i - 12.3) * -22.04);
        b[i] = std::complex<Z>((i - 10) * 39.66, (i + 18.3) * 21.04);
    }

    c = a / b;

    for (size_t i = 0; i < 1024; ++i) {
        REQUIRE_EQUALS_APPROX(c[i].real(), (a[i] / b[i]).real());
        REQUIRE_EQUALS_APPROX(c[i].imag(), (a[i] / b[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("complex/std/9", "[complex]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 1, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_matrix<std::complex<Z>, 1, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};
    etl::fast_matrix<std::complex<Z>, 1, 3> c;

    c(0) = a(0) >> b(0);

    REQUIRE_EQUALS(c(0)[0], a(0)[0] * b(0)[0]);
    REQUIRE_EQUALS(c(0)[1], a(0)[1] * b(0)[1]);
    REQUIRE_EQUALS(c(0)[2], a(0)[2] * b(0)[2]);
}

GEMM_TEST_CASE("complex/std/10", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix<std::complex<Z>, 2, 3> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};
    etl::fast_matrix<std::complex<Z>, 3, 2> b = {CZ(1, 1), CZ(2, 2), CZ(3, 2), CZ(1, 0), CZ(1, -1), CZ(2, 2)};
    etl::fast_matrix<std::complex<Z>, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0).real(), 3.0);
    REQUIRE_EQUALS(c(0, 0).imag(), -7.0);
    REQUIRE_EQUALS(c(0, 1).real(), -4.0);
    REQUIRE_EQUALS(c(0, 1).imag(), 12.0);
    REQUIRE_EQUALS(c(1, 0).real(), 5.0);
    REQUIRE_EQUALS(c(1, 0).imag(), 5.0);
    REQUIRE_EQUALS(c(1, 1).real(), 1.0);
    REQUIRE_EQUALS(c(1, 1).imag(), 9.0);
}

GEMV_TEST_CASE("complex/std/11", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix<std::complex<Z>, 2, 3> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};
    etl::fast_vector<std::complex<Z>, 3> b    = {CZ(1, 1), CZ(-3, -3), CZ(5, 0.1)};
    etl::fast_matrix<std::complex<Z>, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), Z(9.7));
    REQUIRE_EQUALS_APPROX(c(0).imag(), Z(29.2));
    REQUIRE_EQUALS_APPROX(c(1).real(), Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag(), Z(4.2));
}

GEVM_TEST_CASE("complex/std/12", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};
    etl::fast_vector<std::complex<Z>, 3> b    = {CZ(1, 1), CZ(-3, -3), CZ(5, 0.1)};
    etl::fast_matrix<std::complex<Z>, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), Z(7.9));
    REQUIRE_EQUALS_APPROX(c(0).imag(), Z(-7.9));
    REQUIRE_EQUALS_APPROX(c(1).real(), Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag(), Z(6.2));
}

TEMPLATE_TEST_CASE_2("complex/std/13", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};

    a >>= b;

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(-0.99));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(1.32));
    REQUIRE_EQUALS_APPROX(a[1].real(), Z(1.5));
    REQUIRE_EQUALS_APPROX(a[1].imag(), Z(3.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), Z(-0.375));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(0.25));
}

TEMPLATE_TEST_CASE_2("complex/std/14", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};
    etl::fast_vector<std::complex<Z>, 3> b = {std::complex<Z>(0.33, 0.66), std::complex<Z>(-1.5, 0.0), std::complex<Z>(0.5, 0.75)};

    a = a >> b;

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(-0.99));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(1.32));
    REQUIRE_EQUALS_APPROX(a[1].real(), Z(1.5));
    REQUIRE_EQUALS_APPROX(a[1].imag(), Z(3.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), Z(-0.375));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(0.25));
}

TEMPLATE_TEST_CASE_2("complex/std/15", "[complex]", Z, float, double) {
    etl::fast_vector<std::complex<Z>, 3> a = {std::complex<Z>(1.0, 2.0), std::complex<Z>(-1.0, -2.0), std::complex<Z>(0.0, 0.5)};

    a >>= a;

    REQUIRE_EQUALS_APPROX(a[0].real(), Z(-3.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), Z(4.0));
    REQUIRE_EQUALS_APPROX(a[1].real(), Z(-3.0));
    REQUIRE_EQUALS_APPROX(a[1].imag(), Z(4.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), Z(-0.25));
    REQUIRE_EQUALS_APPROX(a[2].imag(), Z(0.0));
}

TEMPLATE_TEST_CASE_2("complex/etl/13", "[complex]", Z, float, double) {
    etl::fast_vector<etl::complex<Z>, 3> a = {etl::complex<Z>(1.0, 2.0), etl::complex<Z>(-1.0, -2.0), etl::complex<Z>(0.0, 0.5)};
    etl::fast_vector<etl::complex<Z>, 3> b = {etl::complex<Z>(0.33, 0.66), etl::complex<Z>(-1.5, 0.0), etl::complex<Z>(0.5, 0.75)};

    a >>= b;

    REQUIRE_EQUALS_APPROX(a[0].real, Z(-0.99));
    REQUIRE_EQUALS_APPROX(a[0].imag, Z(1.32));
    REQUIRE_EQUALS_APPROX(a[1].real, Z(1.5));
    REQUIRE_EQUALS_APPROX(a[1].imag, Z(3.0));
    REQUIRE_EQUALS_APPROX(a[2].real, Z(-0.375));
    REQUIRE_EQUALS_APPROX(a[2].imag, Z(0.25));
}

TEMPLATE_TEST_CASE_2("complex/etl/14", "[complex]", Z, float, double) {
    etl::fast_vector<etl::complex<Z>, 3> a = {etl::complex<Z>(1.0, 2.0), etl::complex<Z>(-1.0, -2.0), etl::complex<Z>(0.0, 0.5)};
    etl::fast_vector<etl::complex<Z>, 3> b = {etl::complex<Z>(0.33, 0.66), etl::complex<Z>(-1.5, 0.0), etl::complex<Z>(0.5, 0.75)};

    a = a >> b;

    REQUIRE_EQUALS_APPROX(a[0].real, Z(-0.99));
    REQUIRE_EQUALS_APPROX(a[0].imag, Z(1.32));
    REQUIRE_EQUALS_APPROX(a[1].real, Z(1.5));
    REQUIRE_EQUALS_APPROX(a[1].imag, Z(3.0));
    REQUIRE_EQUALS_APPROX(a[2].real, Z(-0.375));
    REQUIRE_EQUALS_APPROX(a[2].imag, Z(0.25));
}

TEMPLATE_TEST_CASE_2("complex/etl/9", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 1, 3> a = {etl::complex<Z>(1.0, 2.0), etl::complex<Z>(-1.0, -2.0), etl::complex<Z>(0.0, 0.5)};
    etl::fast_matrix<etl::complex<Z>, 1, 3> b = {etl::complex<Z>(0.33, 0.66), etl::complex<Z>(-1.5, 0.0), etl::complex<Z>(0.5, 0.75)};
    etl::fast_matrix<etl::complex<Z>, 1, 3> c;

    c(0) = a(0) >> b(0);

    REQUIRE_EQUALS(c(0)[0], a(0)[0] * b(0)[0]);
    REQUIRE_EQUALS(c(0)[1], a(0)[1] * b(0)[1]);
    REQUIRE_EQUALS(c(0)[2], a(0)[2] * b(0)[2]);
}

GEMV_TEST_CASE("complex/etl/11", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix<etl::complex<Z>, 2, 3> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};
    etl::fast_vector<etl::complex<Z>, 3> b    = {ECZ(1, 1), ECZ(-3, -3), ECZ(5, 0.1)};
    etl::fast_matrix<etl::complex<Z>, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0).real, Z(9.7));
    REQUIRE_EQUALS_APPROX(c(0).imag, Z(29.2));
    REQUIRE_EQUALS_APPROX(c(1).real, Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag, Z(4.2));
}

GEVM_TEST_CASE("complex/etl/12", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};
    etl::fast_vector<etl::complex<Z>, 3> b    = {ECZ(1, 1), ECZ(-3, -3), ECZ(5, 0.1)};
    etl::fast_matrix<etl::complex<Z>, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS_APPROX(c(0).real, Z(7.9));
    REQUIRE_EQUALS_APPROX(c(0).imag, Z(-7.9));
    REQUIRE_EQUALS_APPROX(c(1).real, Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag, Z(6.2));
}

TEMPLATE_TEST_CASE_2("complex/real/1", "[complex]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = etl::real(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 2);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/real/2", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = etl::real(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 2);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/real/3", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = a.real();

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 2);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/imag/1", "[complex]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = etl::imag(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 3);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/imag/2", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = etl::imag(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 3);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/imag/3", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};

    etl::fast_matrix<Z, 3, 2> b;

    b = a.imag();

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), -2);
    REQUIRE_EQUALS(b(1, 0), 3);
    REQUIRE_EQUALS(b(1, 1), 0);
    REQUIRE_EQUALS(b(2, 0), 1);
    REQUIRE_EQUALS(b(2, 1), 2);
}

TEMPLATE_TEST_CASE_2("complex/conj/1", "[complex]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};
    etl::fast_matrix<std::complex<Z>, 3, 2> b;

    b = etl::conj(a);

    REQUIRE_EQUALS(b(0, 0).real(), 1);
    REQUIRE_EQUALS(b(0, 0).imag(), -1);
    REQUIRE_EQUALS(b(0, 1).real(), -2);
    REQUIRE_EQUALS(b(0, 1).imag(), 2);

    REQUIRE_EQUALS(b(1, 0).real(), 2);
    REQUIRE_EQUALS(b(1, 0).imag(), -3);
    REQUIRE_EQUALS(b(1, 1).real(), 0);
    REQUIRE_EQUALS(b(1, 1).imag(), 0);

    REQUIRE_EQUALS(b(2, 0).real(), 1);
    REQUIRE_EQUALS(b(2, 0).imag(), -1);
    REQUIRE_EQUALS(b(2, 1).real(), 2);
    REQUIRE_EQUALS(b(2, 1).imag(), -2);
}

TEMPLATE_TEST_CASE_2("complex/conj/2", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};
    etl::fast_matrix<etl::complex<Z>, 3, 2> b;

    b = etl::conj(a);

    REQUIRE_EQUALS(b(0, 0).real, 1);
    REQUIRE_EQUALS(b(0, 0).imag, -1);
    REQUIRE_EQUALS(b(0, 1).real, -2);
    REQUIRE_EQUALS(b(0, 1).imag, 2);

    REQUIRE_EQUALS(b(1, 0).real, 2);
    REQUIRE_EQUALS(b(1, 0).imag, -3);
    REQUIRE_EQUALS(b(1, 1).real, 0);
    REQUIRE_EQUALS(b(1, 1).imag, 0);

    REQUIRE_EQUALS(b(2, 0).real, 1);
    REQUIRE_EQUALS(b(2, 0).imag, -1);
    REQUIRE_EQUALS(b(2, 1).real, 2);
    REQUIRE_EQUALS(b(2, 1).imag, -2);
}

TEMPLATE_TEST_CASE_2("complex/conj/3", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};
    etl::fast_matrix<etl::complex<Z>, 3, 2> b;

    b = a.conj();

    REQUIRE_EQUALS(b(0, 0).real, 1);
    REQUIRE_EQUALS(b(0, 0).imag, -1);
    REQUIRE_EQUALS(b(0, 1).real, -2);
    REQUIRE_EQUALS(b(0, 1).imag, 2);

    REQUIRE_EQUALS(b(1, 0).real, 2);
    REQUIRE_EQUALS(b(1, 0).imag, -3);
    REQUIRE_EQUALS(b(1, 1).real, 0);
    REQUIRE_EQUALS(b(1, 1).imag, 0);

    REQUIRE_EQUALS(b(2, 0).real, 1);
    REQUIRE_EQUALS(b(2, 0).imag, -1);
    REQUIRE_EQUALS(b(2, 1).real, 2);
    REQUIRE_EQUALS(b(2, 1).imag, -2);
}

TEMPLATE_TEST_CASE_2("complex/ctrans/1", "[complex]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(-2, -2), CZ(2, 3), CZ(0, 0), CZ(1, 1), CZ(2, 2)};
    etl::fast_matrix<std::complex<Z>, 2, 3> b;

    b = etl::ctrans(a);

    REQUIRE_EQUALS(b(0, 0).real(), 1);
    REQUIRE_EQUALS(b(0, 0).imag(), -1);
    REQUIRE_EQUALS(b(0, 1).real(), 2);
    REQUIRE_EQUALS(b(0, 1).imag(), -3);
    REQUIRE_EQUALS(b(0, 2).real(), 1);
    REQUIRE_EQUALS(b(0, 2).imag(), -1);

    REQUIRE_EQUALS(b(1, 0).real(), -2);
    REQUIRE_EQUALS(b(1, 0).imag(), 2);
    REQUIRE_EQUALS(b(1, 1).real(), 0);
    REQUIRE_EQUALS(b(1, 1).imag(), 0);
    REQUIRE_EQUALS(b(1, 2).real(), 2);
    REQUIRE_EQUALS(b(1, 2).imag(), -2);
}

TEMPLATE_TEST_CASE_2("complex/ctrans/2", "[complex]", Z, float, double) {
    etl::fast_matrix<etl::complex<Z>, 3, 2> a = {ECZ(1, 1), ECZ(-2, -2), ECZ(2, 3), ECZ(0, 0), ECZ(1, 1), ECZ(2, 2)};
    etl::fast_matrix<etl::complex<Z>, 2, 3> b;

    b = etl::conj_transpose(a);

    REQUIRE_EQUALS(b(0, 0).real, 1);
    REQUIRE_EQUALS(b(0, 0).imag, -1);
    REQUIRE_EQUALS(b(0, 1).real, 2);
    REQUIRE_EQUALS(b(0, 1).imag, -3);
    REQUIRE_EQUALS(b(0, 2).real, 1);
    REQUIRE_EQUALS(b(0, 2).imag, -1);

    REQUIRE_EQUALS(b(1, 0).real, -2);
    REQUIRE_EQUALS(b(1, 0).imag, 2);
    REQUIRE_EQUALS(b(1, 1).real, 0);
    REQUIRE_EQUALS(b(1, 1).imag, 0);
    REQUIRE_EQUALS(b(1, 2).real, 2);
    REQUIRE_EQUALS(b(1, 2).imag, -2);
}
