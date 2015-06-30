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

#include "math.hpp"
#include "compat.hpp"

namespace etl {

using random_engine = std::mt19937_64;

template<typename T>
struct simple_operator : std::true_type {};

template<typename T>
struct plus_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr const bool vectorizable = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs + rhs;
    }

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        const vec_type ymm1(lhs);
        const vec_type ymm2(rhs);
        return vec::add(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "+";
    }
};

template<typename T>
struct minus_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr const bool vectorizable = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs - rhs;
    }

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        const vec_type ymm1(lhs);
        const vec_type ymm2(rhs);
        return vec::sub(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "-";
    }
};

template<typename T>
struct mul_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr const bool vectorizable = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs * rhs;
    }

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        const vec_type ymm1(lhs);
        const vec_type ymm2(rhs);
        return vec::mul(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "*";
    }
};

template<typename T>
struct div_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr const bool vectorizable = true;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs / rhs;
    }

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        const vec_type ymm1(lhs);
        const vec_type ymm2(rhs);
        return vec::div(ymm1, ymm2);
    }

    static std::string desc() noexcept {
        return "/";
    }
};

template<typename T>
struct mod_binary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs % rhs;
    }

    static std::string desc() noexcept {
        return "%";
    }
};

template<typename T, typename E>
struct ranged_noise_binary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x, E value){
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);
        static auto noise = std::bind(normal_distribution, rand_engine);

        if(x == 0.0 || x == value){
            return x;
        } else {
            return x + noise();
        }
    }

    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

template<typename T, typename E>
struct max_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::max(x, value);
    }

#ifdef __INTEL_COMPILER
    static constexpr const bool vectorizable = true;

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        return vec::max(lhs, rhs);
    }
#else
    static constexpr const bool vectorizable = false;
#endif

    static std::string desc() noexcept {
        return "max";
    }
};

template<typename T, typename E>
struct min_binary_op {
    using vec_type = intrinsic_type<T>;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::min(x, value);
    }

#ifdef __INTEL_COMPILER
    static constexpr const bool vectorizable = true;

    static cpp14_constexpr vec_type load(const vec_type& lhs, const vec_type& rhs) noexcept {
        return vec::min(lhs, rhs);
    }
#else
    static constexpr const bool vectorizable = false;
#endif

    static std::string desc() noexcept {
        return "min";
    }
};

template<typename T, typename S>
struct min_scalar_op {
    using vec_type = intrinsic_type<T>;

    S s;
    min_scalar_op(S s) : s(s) {}

    constexpr T apply(const T& x) const noexcept {
        return std::min(x, s);
    }

#ifdef __INTEL_COMPILER
    static constexpr const bool vectorizable = true;

    cpp14_constexpr vec_type load(const vec_type& lhs) const noexcept {
        return vec::min(lhs, vec::set(s));
    }
#else
    static constexpr const bool vectorizable = false;
#endif

    static std::string desc() noexcept {
        return "min";
    }
};

template<typename T, typename S>
struct max_scalar_op {
    using vec_type = intrinsic_type<T>;

    S s;
    max_scalar_op(S s) : s(s) {}

    constexpr T apply(const T& x) const noexcept {
        return std::max(x, s);
    }

#ifdef __INTEL_COMPILER
    static constexpr const bool vectorizable = true;

    cpp14_constexpr vec_type load(const vec_type& lhs) const noexcept {
        return vec::max(lhs, vec::set(s));
    }
#else
    static constexpr const bool vectorizable = false;
#endif

    static std::string desc() noexcept {
        return "max";
    }
};

template<typename T, typename S>
struct clip_scalar_op {
    using vec_type = intrinsic_type<T>;

    S min;
    S max;
    clip_scalar_op(S min, S max) : min(min), max(max) {}

    constexpr T apply(const T& x) const noexcept {
        return std::min(std::max(x, min), max);
    }

#ifdef __INTEL_COMPILER
    static constexpr const bool vectorizable = true;

    cpp14_constexpr vec_type load(const vec_type& lhs) const noexcept {
        return vec::min(vec::max(lhs, vec::set(min)), vec::set(max));
    }
#else
    static constexpr const bool vectorizable = false;
#endif

    static std::string desc() noexcept {
        return "clip";
    }
};

template<typename T, typename E>
struct pow_binary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x, E value) noexcept {
        return std::pow(x, value);
    }

    static std::string desc() noexcept {
        return "pow";
    }
};

template<typename T, typename E>
struct one_if_binary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x, E value) noexcept {
        return 1.0 ? x == value : 0.0;
    }

    static std::string desc() noexcept {
        return "one_if";
    }
};

//Define operators which are no simple c++ operators (+,-,*,...)
//These operators will be displayed differently

template<typename T, typename E>
struct simple_operator<ranged_noise_binary_op<T, E>> : std::false_type {};

template<typename T, typename E>
struct simple_operator<max_binary_op<T, E>> : std::false_type {};

template<typename T, typename E>
struct simple_operator<min_binary_op<T, E>> : std::false_type {};

template<typename T, typename E>
struct simple_operator<pow_binary_op<T, E>> : std::false_type {};

template<typename T, typename E>
struct simple_operator<one_if_binary_op<T, E>> : std::false_type {};

} //end of namespace etl
