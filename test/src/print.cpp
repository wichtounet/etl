//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// stream

namespace {

template <typename T>
std::string to_stream_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

} //end of anonymous namespace

ETL_TEST_CASE("print/expr/1", "[print][stream]") {
    etl::fast_vector<double, 3> a;
    etl::dyn_vector<double> b(3);
    etl::fast_matrix<double, 3, 4> c;
    etl::dyn_matrix<double> d(3, 4);

    REQUIRE_EQUALS(to_stream_string(a), "V[3]");
    REQUIRE_EQUALS(to_stream_string(b), "V[3]");
    REQUIRE_EQUALS(to_stream_string(c), "M[3,4]");
    REQUIRE_EQUALS(to_stream_string(d), "M[3,4]");
}

ETL_TEST_CASE("print/expr/2", "[print][stream]") {
    etl::fast_matrix<double, 3, 4> a;

    REQUIRE_EQUALS(to_stream_string(a + a), "(M[3,4] + M[3,4])");
    REQUIRE_EQUALS(to_stream_string(a + a + a), "((M[3,4] + M[3,4]) + M[3,4])");
    REQUIRE_EQUALS(to_stream_string(a / a >> a), "((M[3,4] / M[3,4]) * M[3,4])");
    REQUIRE_EQUALS(to_stream_string(etl::log(a) + etl::abs(a)), "(log(M[3,4]) + abs(M[3,4]))");
    REQUIRE_EQUALS(to_stream_string(a(1) + etl::abs(a(0) - a(1))), "(sub(M[3,4], 1) + abs((sub(M[3,4], 0) - sub(M[3,4], 1))))");
}

// to_octave

ETL_TEST_CASE("to_octave/fast_vector", "to_octave") {
    etl::fast_vector<double, 3> test_vector({1.0, -2.0, 3.0});

    REQUIRE_EQUALS(to_octave(test_vector), "[1.000000,-2.000000,3.000000]");
}

ETL_TEST_CASE("to_octave/fast_matrix", "to_octave") {
    etl::fast_matrix<double, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE_EQUALS(to_octave(test_matrix), "[1.000000,-2.000000;3.000000,0.500000;0.000000,-1.000000]");
}

ETL_TEST_CASE("to_octave/dyn_vector", "to_octave") {
    etl::dyn_vector<double> test_vector({1.0, -2.0, 3.0});

    REQUIRE_EQUALS(to_octave(test_vector), "[1.000000,-2.000000,3.000000]");
}

ETL_TEST_CASE("to_octave/dyn_matrix", "to_octave") {
    etl::dyn_matrix<double> test_matrix(3, 2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    REQUIRE_EQUALS(to_octave(test_matrix), "[1.000000,-2.000000;3.000000,0.500000;0.000000,-1.000000]");
}

// to_string

ETL_TEST_CASE("to_string/fast_vector", "to_string") {
    etl::fast_vector<double, 3> test_vector({1.0, -2.0, 3.0});

    REQUIRE_EQUALS(to_string(test_vector), "[1.000000,-2.000000,3.000000]");
}

ETL_TEST_CASE("to_string/fast_matrix", "to_string") {
    etl::fast_matrix<double, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE_EQUALS(to_string(test_matrix), "[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]");
}

ETL_TEST_CASE("to_string/fast_matrix_3d", "to_string") {
    etl::fast_matrix<double, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE_EQUALS(to_string(test_matrix), "[[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]\n[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]]");
}

ETL_TEST_CASE("to_string/dyn_vector", "to_string") {
    etl::dyn_vector<double> test_vector({1.0, -2.0, 3.0});

    REQUIRE_EQUALS(to_string(test_vector), "[1.000000,-2.000000,3.000000]");
}

ETL_TEST_CASE("to_string/dyn_matrix", "to_string") {
    etl::dyn_matrix<double> test_matrix(3, 2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    REQUIRE_EQUALS(to_string(test_matrix), "[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]");
}
