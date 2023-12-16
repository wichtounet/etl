//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

// 2D batch_hint
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        batch_hint_2d_policy,
        "R = batch_hint_2d(gamma >> hint) (s) [batch_hint][s]",
        [](auto B, auto I){ return std::make_tuple(smat(B, I), svec(I), smat(B, I)); },
        [](smat& lhs, svec& gamma, smat& input){ lhs = batch_hint(gamma >> input); }
        );

    CPM_TWO_PASS_NS_P(
        batch_hint_2d_policy,
        "R = batch_hint_2d((gamma >> hint) + beta) (s) [batch_hint][s]",
        [](auto B, auto I){ return std::make_tuple(smat(B, I), svec(I), svec(I), smat(B, I)); },
        [](smat& lhs, svec& gamma, svec& beta, smat& input){ lhs = batch_hint((gamma >> input) + beta); }
        );

    CPM_TWO_PASS_NS_P(
        batch_hint_2d_policy,
        "R = batch_hint_2d(gamma >> (hint - beta) (s) [batch_hint][s]",
        [](auto B, auto I){ return std::make_tuple(smat(B, I), svec(I), svec(I), smat(B, I)); },
        [](smat& lhs, svec& gamma, svec& beta, smat& input){ lhs = batch_hint(gamma >> (input - beta)); }
        );
}

// 4D batch_hint
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        batch_hint_4d_policy,
        "R = batch_hint_4d(gamma >> hint) (s) [batch_hint][s]",
        [](auto B, auto K, auto M, auto N){ return std::make_tuple(smat4(B, K, M, N), svec(K), smat4(B, K, M, N)); },
        [](smat4& lhs, svec& gamma, smat4& input){ lhs = batch_hint(gamma >> input); }
        );

    CPM_TWO_PASS_NS_P(
        batch_hint_4d_policy,
        "R = batch_hint_4d((gamma >> hint) + beta) (s) [batch_hint][s]",
        [](auto B, auto K, auto M, auto N){ return std::make_tuple(smat4(B, K, M, N), svec(K), svec(K), smat4(B, K, M, N)); },
        [](smat4& lhs, svec& gamma, svec& beta, smat4& input){ lhs = batch_hint((gamma >> input) + beta); }
        );

    CPM_TWO_PASS_NS_P(
        batch_hint_4d_policy,
        "R = batch_hint_4d(gamma >> (hint - beta) (s) [batch_hint][s]",
        [](auto B, auto K, auto M, auto N){ return std::make_tuple(smat4(B, K, M, N), svec(K), svec(K), smat4(B, K, M, N)); },
        [](smat4& lhs, svec& gamma, svec& beta, smat4& input){ lhs = batch_hint(gamma >> (input - beta)); }
        );
}
