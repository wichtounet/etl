//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// conv_4d_valid

TEMPLATE_TEST_CASE_2("opaque/1", "[opaque]", T, double, float) {
    etl::fast_matrix<T, 10, 2, 5, 5> I;
    etl::fast_matrix<T, 12, 2, 3, 3> K;
    etl::fast_matrix<T, 10, 12, 3, 3> C;

    auto i_direct = I.direct();
    auto k_direct = K.direct();
    auto c_direct = C.direct();

    REQUIRE_EQUALS(i_direct.memory_start(), I.memory_start());
    REQUIRE_EQUALS(k_direct.memory_start(), K.memory_start());
    REQUIRE_EQUALS(c_direct.memory_start(), C.memory_start());
}
