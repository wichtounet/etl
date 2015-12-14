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
struct abs_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::abs(x);
    }

    static std::string desc() noexcept {
        return "abs";
    }
};

template <typename T>
struct log_unary_op {
    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;

    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr T apply(const T& x) {
        return std::log(x);
    }


#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::log(x);
    }
#endif

    static std::string desc() noexcept {
        return "log";
    }
};

template <typename T>
struct sqrt_unary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = !is_complex_t<T>::value;

    static constexpr T apply(const T& x) {
        return std::sqrt(x);
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::sqrt(x);
    }

    static std::string desc() noexcept {
        return "sqrt";
    }
};

template <typename T>
struct exp_unary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value;

    static constexpr T apply(const T& x) {
        return std::exp(x);
    }

#ifdef __INTEL_COMPILER
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::exp(x);
    }
#endif

    static std::string desc() noexcept {
        return "exp";
    }
};

template <typename T>
struct sign_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return math::sign(x);
    }

    static std::string desc() noexcept {
        return "sign";
    }
};

template <typename T>
struct sigmoid_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) {
        return math::logistic_sigmoid(x);
    }

    static std::string desc() noexcept {
        return "sigmoid";
    }
};

template <typename T>
struct softplus_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) {
        return math::softplus(x);
    }

    static std::string desc() noexcept {
        return "softplus";
    }
};

template <typename T>
struct minus_unary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = !is_complex_t<T>::value;

    static constexpr T apply(const T& x) noexcept {
        return -x;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::minus(x);
    }

    static std::string desc() noexcept {
        return "-";
    }
};

template <typename T>
struct plus_unary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = true;

    static constexpr T apply(const T& x) noexcept {
        return +x;
    }

    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& x) noexcept {
        return x;
    }

    static std::string desc() noexcept {
        return "+";
    }
};

template <typename T>
struct fast_sigmoid_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& v) {
        auto x = 0.5 * v;

        T z;
        if (x >= 0) {
            if (x < 1.7) {
                z = (1.5 * x / (1 + x));
            } else if (x < 3) {
                z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7));
            } else {
                z = 0.99505475368673;
            }
        } else {
            auto xx = -x;
            if (xx < 1.7) {
                z = (1.5 * xx / (1 + xx));
            } else if (xx < 3) {
                z = (0.935409070603099 + 0.0458812946797165 * (xx - 1.7));
            } else {
                z = 0.99505475368673;
            }
            z = -z;
        }

        return 0.5 * (z + 1.0);
    }

    static std::string desc() noexcept {
        return "fast_sigmoid";
    }
};

template <typename T>
struct tan_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::tan(x);
    }

    static std::string desc() noexcept {
        return "tan";
    }
};

template <typename T>
struct cos_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::cos(x);
    }

    static std::string desc() noexcept {
        return "cos";
    }
};

template <typename T>
struct sin_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::sin(x);
    }

    static std::string desc() noexcept {
        return "sin";
    }
};

template <typename T>
struct tanh_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::tanh(x);
    }

    static std::string desc() noexcept {
        return "tanh";
    }
};

template <typename T>
struct cosh_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::cosh(x);
    }

    static std::string desc() noexcept {
        return "cosh";
    }
};

template <typename T>
struct sinh_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return std::sinh(x);
    }

    static std::string desc() noexcept {
        return "sinh";
    }
};

template <typename T>
struct real_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr value_t<T> apply(const T& x) noexcept {
        return get_real(x);
    }

    static std::string desc() noexcept {
        return "real";
    }
};

template <typename T>
struct imag_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr value_t<T> apply(const T& x) noexcept {
        return get_imag(x);
    }

    static std::string desc() noexcept {
        return "imag";
    }
};

template <typename T>
struct conj_unary_op {
    static constexpr const bool vectorizable = false;

    static constexpr T apply(const T& x) noexcept {
        return get_conj(x);
    }

    static std::string desc() noexcept {
        return "conj";
    }
};

template <typename T>
struct relu_derivative_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    static std::string desc() noexcept {
        return "relu_derivative_op";
    }
};

template <typename T>
struct bernoulli_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    static std::string desc() noexcept {
        return "bernoulli";
    }
};

template <typename T>
struct reverse_bernoulli_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 0.0 : 1.0;
    }

    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

template <typename T>
struct uniform_noise_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> real_distribution(0.0, 1.0);

        return x + real_distribution(rand_engine);
    }

    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

template <typename T>
struct normal_noise_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);

        return x + normal_distribution(rand_engine);
    }

    static std::string desc() noexcept {
        return "normal_noise";
    }
};

template <typename T>
struct logistic_noise_unary_op {
    static constexpr const bool vectorizable = false;

    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));

        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
    }

    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

} //end of namespace etl
