//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp<2x2>(d=2) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d / 2,d / 2)); },
        [](smat& a, smat& r){ r = etl::ml::max_pool_forward<2, 2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp<2x2>(d=3) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat3(100, d, d), smat3(100, d / 2,d / 2)); },
        [](smat3& a, smat3& r){ r = etl::ml::max_pool_forward<2, 2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp<2x2>(d=4) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat4(100, 10, d, d), smat4(100, 10, d / 2, d / 2)); },
        [](smat4& a, smat4& r){ r = etl::ml::max_pool_forward<2, 2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp(d=2, 2, 2) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d / 2,d / 2)); },
        [](smat& a, smat& r){ r = etl::ml::max_pool_forward(a, 2, 2); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp(d=3, 2, 2) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat3(100, d, d), smat3(100, d / 2,d / 2)); },
        [](smat3& a, smat3& r){ r = etl::ml::max_pool_forward(a, 2, 2); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        mp_policy,
        "mp(d=4, 2, 2) (s) [mp][s]",
        [](size_t d){ return std::make_tuple(smat4(100, 10, d, d), smat4(100, 10, d / 2, d / 2)); },
        [](smat4& a, smat4& r){ r = etl::ml::max_pool_forward(a, 2, 2); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=2) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d,d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

#ifdef ETL_EXTENDED_BENCH
    CPM_TWO_PASS_NS_P(
        pmp_policy_3,
        "pmp_h_3(c=2) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat3(50UL, d, d), smat3(50UL, d,d)); },
        [](smat3& a, smat3& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](size_t d){ return 50 * 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy_3,
        "pmp_h_4(c=2) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat4(50UL, 50UL, d, d), smat4(50UL, 50UL, d,d)); },
        [](smat4& a, smat4& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](size_t d){ return 50 * 50 * 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "dyn_pmp_h(c=2) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d, d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h(a, 2, 2); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );
#endif

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=2) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d/2,d/2)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_p<2,2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=4) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d,d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h<4,4>(a); },
        [](size_t d){ return 2 * d * d * 4 * 4; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=4) (s) [pmp][s]",
        [](size_t d){ return std::make_tuple(smat(d, d), smat(d/4,d/4)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_p<4,4>(a); },
        [](size_t d){ return 2 * d * d * 4 * 4; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=2) (d) [pmp][s]",
        [](size_t d){ return std::make_tuple(dmat(d, d), dmat(d,d)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=2) (d) [pmp][s]",
        [](size_t d){ return std::make_tuple(dmat(d, d), dmat(d/2,d/2)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_p<2,2>(a); },
        [](size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=4) (d) [pmp][s]",
        [](size_t d){ return std::make_tuple(dmat(d, d), dmat(d,d)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_h<4,4>(a); },
        [](size_t d){ return 2 * d * d * 4 * 4; }
        );
CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=4) (d) [pmp][s]",
        [](size_t d){ return std::make_tuple(dmat(d, d), dmat(d/4,d/4)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_p<4,4>(a); },
        [](size_t d){ return 2 * d * d * 4 * 4; }
        );
}
