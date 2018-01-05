//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

//Bench trigonometric function
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = cos(a) (s) [std][cos][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = cos(a); }
        );

    CPM_TWO_PASS_NS(
        "r = sin(a) (s) [std][sin][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = sin(a); }
        );

    CPM_TWO_PASS_NS(
        "r = tan(a) (s) [std][tan][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = tan(a); }
        );
}

//Bench hyperbolic function
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = cosh(a) (s) [std][cosh][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = cosh(a); }
        );

    CPM_TWO_PASS_NS(
        "r = sinh(a) (s) [std][sinh][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = sinh(a); }
        );

    CPM_TWO_PASS_NS(
        "r = tanh(a) (s) [std][tanh][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = tanh(a); }
        );
}
