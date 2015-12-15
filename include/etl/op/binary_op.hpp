//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <random>
#include <functional>
#include <ctime>

#include "etl/math.hpp"

namespace etl {

using random_engine = std::mt19937_64;

template <typename T>
struct simple_operator : std::true_type {};

template <typename T>
struct plus_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = true;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs + rhs;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::add(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "+";
    }
};

template <typename T>
struct minus_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = true;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs - rhs;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::sub(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "-";
    }
};

template <typename T>
struct mul_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = vector_mode == vector_mode_t::AVX512 ? !is_complex_t<T>::value : true ;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs * rhs;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::template mul<is_complex_t<T>::value>(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "*";
    }
};

template <typename T>
struct div_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = vector_mode == vector_mode_t::AVX512 ? !is_complex_t<T>::value : true ;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs / rhs;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::template div<is_complex_t<T>::value>(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "/";
    }
};

template <typename T>
struct mod_binary_op {
    static constexpr const bool vectorizable = false;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs % rhs;
    }

    static std::string desc() noexcept {
        return "%";
    }
};

template <typename T, typename E>
struct ranged_noise_binary_op {
    static constexpr const bool vectorizable = false;
    static constexpr const bool linear       = true;

    static T apply(const T& x, E value) {
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);
        static auto noise = std::bind(normal_distribution, rand_engine);

        if (x == 0.0 || x == value) {
            return x;
        } else {
            return x + noise();
        }
    }

    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

template <typename T, typename E>
struct max_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::max(x, value);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::max(lhs, rhs);
    }
#endif

    static std::string desc() noexcept {
        return "max";
    }
};

template <typename T, typename E>
struct min_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::min(x, value);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::min(lhs, rhs);
    }
#endif

    static std::string desc() noexcept {
        return "min";
    }
};

template <typename T, typename S>
struct min_scalar_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;
    static constexpr const bool linear       = true;

    S s;
    explicit min_scalar_op(S s)
            : s(s) {}

    constexpr T apply(const T& x) const noexcept {
        return std::min(x, s);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::min(lhs, V::set(s));
    }
#endif

    static std::string desc() noexcept {
        return "min";
    }
};

template <typename T, typename S>
struct max_scalar_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;
    static constexpr const bool linear       = true;

    S s;
    explicit max_scalar_op(S s)
            : s(s) {}

    constexpr T apply(const T& x) const noexcept {
        return std::max(x, s);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::max(lhs, V::set(s));
    }
#endif

    static std::string desc() noexcept {
        return "max";
    }
};

template <typename T, typename S>
struct clip_scalar_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;
    static constexpr const bool linear       = true;

    S min;
    S max;
    clip_scalar_op(S min, S max)
            : min(min), max(max) {}

    constexpr T apply(const T& x) const noexcept {
        return std::min(std::max(x, min), max);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::min(V::max(lhs, V::set(min)), V::set(max));
    }
#endif

    static std::string desc() noexcept {
        return "clip";
    }
};

template <typename T, typename E>
struct pow_binary_op {
    static constexpr const bool vectorizable = false;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::pow(x, value);
    }

    static std::string desc() noexcept {
        return "pow";
    }
};

template <typename T, typename E>
struct one_if_binary_op {
    static constexpr const bool vectorizable = false;
    static constexpr const bool linear       = true;

    static constexpr T apply(const T& x, E value) noexcept {
        return 1.0 ? x == value : 0.0;
    }

    static std::string desc() noexcept {
        return "one_if";
    }
};

//Define operators which are no simple c++ operators (+,-,*,...)
//These operators will be displayed differently

template <typename T, typename E>
struct simple_operator<ranged_noise_binary_op<T, E>> : std::false_type {};

template <typename T, typename E>
struct simple_operator<max_binary_op<T, E>> : std::false_type {};

template <typename T, typename E>
struct simple_operator<min_binary_op<T, E>> : std::false_type {};

template <typename T, typename E>
struct simple_operator<pow_binary_op<T, E>> : std::false_type {};

template <typename T, typename E>
struct simple_operator<one_if_binary_op<T, E>> : std::false_type {};

} //end of namespace etl
