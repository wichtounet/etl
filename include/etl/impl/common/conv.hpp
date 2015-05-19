//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_COMMON_CONVOLUTION_HPP
#define ETL_IMPL_COMMON_CONVOLUTION_HPP

namespace etl {

namespace impl {

namespace common {

template<typename T>
void left_same_kernel(const T* in, const std::size_t /*n*/, const T* kernel, std::size_t m, T* out){
    std::size_t left = (m - 1) / 2;
    std::size_t right = m / 2;

    //Left invalid part
    for(std::size_t j = 0 ; j < left ; ++j){
        T temp = 0.0;

        for(std::size_t l = 0 ; l <= j + right; ++l){
            temp += in[l] * kernel[j - l + right];
        }

        out[j] = temp;
    }
}

template<typename T>
void right_same_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t left = (m - 1) / 2;
    std::size_t right = m / 2;

    //Right invalid part
    for(std::size_t j = n - right ; j < n; ++j){
        T temp = 0.0;

        std::size_t hi = std::min<int>(n - 1, j + right);
        for(std::size_t l = j - left ; l <= hi; ++l){
            temp += in[l] * kernel[j - l + m / 2];
        }

        out[j] = temp;
    }
}

template<typename T>
void left_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t left = m - 1;

    //Left invalid part
    for(std::size_t i = 0; i < left; ++i) {
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for(std::size_t j = 0; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

template<typename T>
void right_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t right = m - 1;

    auto c = n + m - 1;

    //Right invalid part
    for(std::size_t i = c - right; i < c; ++i) {
        const auto lo = i >= m - 1 ? i - (m - 1) : 0;
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for(std::size_t j = lo; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

} //end of namespace common
} //end of namespace impl
} //end of namespace etl

#endif
