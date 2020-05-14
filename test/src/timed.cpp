//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

namespace {

bool starts_with(const std::string& str, const std::string& search) {
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

} //end of anonymous namespace

TEMPLATE_TEST_CASE_2("timed/1", "[fast][serial]", Z, float, double) {
    std::stringstream buffer;
    auto* old = std::cout.rdbuf(buffer.rdbuf());

    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = timed(a + a);

    auto text = buffer.str();
    std::cout.rdbuf(old);

    REQUIRE_DIRECT(starts_with(text, "timed(=): (V[3] + V[3]) took "));
    REQUIRE_EQUALS(std::string(text.end() - 3, text.end() - 1), "ns");

    REQUIRE_EQUALS(b[0], 2.0);
}

TEMPLATE_TEST_CASE_2("timed/2", "[dyn][serial]", Z, float, double) {
    std::stringstream buffer;
    auto* old = std::cout.rdbuf(buffer.rdbuf());

    etl::dyn_vector<Z> a(10000);
    etl::dyn_vector<Z> b(10000);

    a = 1.0;
    b = 2.0;

    b = timed(a + b);

    auto text = buffer.str();
    std::cout.rdbuf(old);

    REQUIRE_DIRECT(starts_with(text, "timed(=): (V[10000] + V[10000]) took "));
    REQUIRE_EQUALS(std::string(text.end() - 3, text.end() - 1), "ns");

    REQUIRE_EQUALS(b[0], 3.0);
}

TEMPLATE_TEST_CASE_2("timed/3", "[dyn][serial]", Z, float, double) {
    std::stringstream buffer;
    auto* old = std::cout.rdbuf(buffer.rdbuf());

    etl::dyn_vector<Z> a(10000);
    etl::dyn_vector<Z> b(10000);

    a = 1.0;

    b = etl::timed_res<etl::milliseconds>(a + a);

    auto text = buffer.str();
    std::cout.rdbuf(old);

    REQUIRE_DIRECT(starts_with(text, "timed(=): (V[10000] + V[10000]) took "));
    REQUIRE_EQUALS(std::string(text.end() - 3, text.end() - 1), "ms");

    REQUIRE_EQUALS(b[0], 2.0);
}
