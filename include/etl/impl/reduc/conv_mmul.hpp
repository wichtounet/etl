//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

namespace etl {

namespace impl {

namespace reduc {

template<typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv){
    conv = row(
        mul(
            reshape(kernel, 1, dim<0>(kernel)),
            convmtx(input, dim<0>(kernel))
        )
        , 0
    );
}

template<typename I, typename K, typename C, cpp_enable_if(all_row_major<I,K,C>::value)>
void conv2_full(const I& input, const K& kernel, C&& conv){
    conv = transpose(reshape(
        mul(
            convmtx2(input, etl::dim<0>(kernel), etl::dim<1>(kernel)),
            reshape(transpose(kernel), etl::size(kernel), 1)
        ),
        etl::dim<1>(input) + etl::dim<1>(kernel) - 1, etl::dim<0>(input) + etl::dim<0>(kernel) - 1)
    );
}

} //end of namespace reduc
} //end of namespace impl
} //end of namespace etl
