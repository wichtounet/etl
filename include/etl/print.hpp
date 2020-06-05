//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <string>

namespace etl {

/*!
 * \brief Construct a textual representation of the matrix contents
 * \param m The expression to transform
 * \return a string representing the contents of the expression
 */
template <typename T>
std::string to_string(T&& m) {
    if constexpr (decay_traits<T>::dimensions() > 1) {
        etl::force(m);

        std::string v = "[";
        for (size_t i = 0; i < etl::dim<0>(m); ++i) {
            v += to_string(sub(m, i));

            if (i < etl::dim<0>(m) - 1) {
                v += "\n";
            }
        }
        v += "]";
        return v;
    } else {
        return to_octave(m);
    }
}

/*!
 * \brief Construct a textual representation of the matrix contents, following the octave format
 * \param m The expression to transform
 * \return a string representing the contents of the expression
 */
template <bool Sub = false, typename T>
std::string to_octave(T&& m) {
    etl::force(m);

    std::string v;

    if (!Sub) {
        v = "[";
    }

    if constexpr (decay_traits<T>::dimensions() > 1) {
        for (size_t i = 0; i < etl::dim<0>(m); ++i) {
            v += to_octave<true>(sub(m, i));

            if (i < etl::dim<0>(m) - 1) {
                v += ";";
            }
        }

        if (!Sub) {
            v += "]";
        }
    } else {
        std::string comma;
        for (size_t j = 0; j < etl::dim<0>(m); ++j) {
            v += comma + std::to_string(m(j));
            comma = ",";
        }

        if (!Sub) {
            v += "]";
        }
    }

    return v;
}

} //end of namespace etl
