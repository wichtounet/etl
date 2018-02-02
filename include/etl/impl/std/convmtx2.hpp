//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Direct evaluation of conmtx2
 */
struct convmtx2_direct {
    /*!
     * \brief Apply the convmtx2 to sub into m
     * \param sub The sub expression
     * \param m The output matrix
     */
    template <size_t K1, size_t K2, typename A, typename M>
    static void apply(A&& sub, M& m) {
        const size_t i1 = etl::dim<0>(sub);
        const size_t i2 = etl::dim<1>(sub);

        const size_t c_height          = etl::dim<0>(m);
        constexpr size_t c_width = K1 * K2;

        cpp_assert(c_height == ((i1 + K1 - 1) * (i2 + K2 - 1)), "Invalid input height");
        cpp_assert(c_width == etl::dim<1>(m), "Invalid input width");

        const auto max_fill       = c_height - ((i1 + K1 - 1) * ((c_width - 1) / K1) + (c_width - 1) % K1);
        const auto inner_paddings = max_fill - (i1 * i2);
        const auto inner_padding  = inner_paddings / (i2 - 1);

        m = 0;

        for (size_t j = 0; j < c_width; ++j) {
            auto top_padding    = (i1 + K1 - 1) * (j / K1) + j % K1;
            auto bottom_padding = top_padding + (i1 * i2) + inner_paddings;

            for (size_t i = top_padding; i < bottom_padding; ++i) {
                auto inner = i - top_padding;
                auto block = inner / (i1 + inner_padding);
                auto col   = inner % (i1 + inner_padding);

                if (col < i1) {
                    m(i, j) = sub(col, block);
                }
            }
        }
    }
};

} //end of namespace etl::impl::standard
