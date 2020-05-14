//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include "cpp_utils/algorithm.hpp"

// Trigonometric tests

TEMPLATE_TEST_CASE_2("trigo/tan/1", "[trigo][tan]", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1, 1.0, -0.5, 0.5));
    etl::dyn_matrix<Z> b;

    b = etl::tan(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::tan(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/tan/2", "[trigo][tan]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::tan(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::tan(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::tan(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/tan/3", "[trigo][tan]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::tan(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::tan(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::tan(a[i]).imag);
    }
}

TEMPLATE_TEST_CASE_2("trigo/sin/1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b;

    b = etl::sin(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::sin(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/sin/2", "[trigo][sin]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::sin(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::sin(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::sin(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/sin/3", "[trigo][sin]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::sin(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::sin(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::sin(a[i]).imag);
    }
}

TEMPLATE_TEST_CASE_2("trigo/cos/1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b;
    b = etl::cos(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::cos(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/cos/2", "[trigo][cos]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::cos(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::cos(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::cos(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/cos/3", "[trigo][cos]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::cos(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::cos(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::cos(a[i]).imag);
    }
}

TEMPLATE_TEST_CASE_2("trigo/tanh/1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1, 0.2, 0.3, -0.2));
    etl::dyn_matrix<Z> b;
    b = etl::tanh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::tanh(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/tanh/2", "[trigo][tanh]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::tanh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::tanh(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::tanh(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/tanh/3", "[trigo][tanh]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::tanh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::tanh(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::tanh(a[i]).imag);
    }
}

TEMPLATE_TEST_CASE_2("trigo/sinh/1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1, 0.2, 0.3, -0.2));
    etl::dyn_matrix<Z> b;
    b = etl::sinh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::sinh(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/sinh/2", "[trigo][sinh]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::sinh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::sinh(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::sinh(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/sinh/3", "[trigo][sinh]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::sinh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::sinh(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::sinh(a[i]).imag);
    }
}

TEMPLATE_TEST_CASE_2("trigo/cosh/1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1, 0.2, 0.3, -0.1));
    etl::dyn_matrix<Z> b;
    b = etl::cosh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::cosh(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/cosh/2", "[trigo][cosh]", Z, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::cosh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real(), std::cosh(a[i]).real());
        REQUIRE_EQUALS_APPROX(b[i].imag(), std::cosh(a[i]).imag());
    }
}

TEMPLATE_TEST_CASE_2("trigo/cosh/3", "[trigo][cosh]", Z, etl::complex<float>, etl::complex<double>) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(Z(1.0, 0.1), Z(2.0, 0.1), Z(-1.0, 0.2), Z(-0.5, 1.0), Z(0.6, 1.0), Z(0.1, 0.4)));
    etl::dyn_matrix<Z> b;

    b = etl::cosh(a);

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i].real, etl::cosh(a[i]).real);
        REQUIRE_EQUALS_APPROX(b[i].imag, etl::cosh(a[i]).imag);
    }
}
