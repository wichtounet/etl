//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_OP_HPP
#define ETL_FAST_OP_HPP

#include <cmath> //For unary operators

namespace etl {

template<typename T>
struct scalar {
    const T value;
    constexpr scalar(T v) : value(v) {}

    constexpr const T operator[](std::size_t) const {
        return value;
    }

    constexpr const T operator()(std::size_t) const {
        return value;
    }

    constexpr const T operator()(std::size_t, std::size_t) const {
        return value;
    }
};

template<typename T>
struct hflip_vector {
    using sub_type = T;

    const T& sub;

    hflip_vector(const T& vec) : sub(vec) {}

    typename T::value_type operator[](std::size_t i) const {
        return sub[size(sub) - 1 - i];
    }
};

template<typename T>
struct hflip_matrix {
    using sub_type = T;

    const T& sub;

    hflip_matrix(const T& vec) : sub(vec) {}

    typename T::value_type operator()(std::size_t i, std::size_t j) const {
        return sub(i, columns(sub) - 1 - j);
    }
};

template<typename T>
struct vflip_vector {
    using sub_type = T;

    const T& sub;

    vflip_vector(const T& vec) : sub(vec) {}

    typename T::value_type operator[](std::size_t i) const {
        return sub[i];
    }
};

template<typename T>
struct plus_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs + rhs;
    }
};

template<typename T>
struct minus_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs - rhs;
    }
};

template<typename T>
struct mul_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs * rhs;
    }
};

template<typename T>
struct div_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs / rhs;
    }
};

template<typename T>
struct mod_binary_op {
    static constexpr T apply(const T& lhs, const T& rhs){
        return lhs % rhs;
    }
};

template<typename T>
struct abs_unary_op {
    static constexpr T apply(const T& x){
        return std::abs(x);
    }
};

template<typename T>
struct log_unary_op {
    static constexpr T apply(const T& x){
        return std::log(x);
    }
};

template<typename T>
struct sign_unary_op {
    static constexpr T apply(const T& x){
        return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0;
    }
};

template<typename T>
struct minus_unary_op {
    static constexpr T apply(const T& x){
        return -x;
    }
};

template<typename T>
struct plus_unary_op {
    static constexpr T apply(const T& x){
        return +x;
    }
};

template<typename T>
struct identity_unary_op {
    static constexpr T apply(const T& x){
        return x;
    }
};

} //end of namespace etl

#endif
