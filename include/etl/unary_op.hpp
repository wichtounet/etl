//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_UNARY_OP_HPP
#define ETL_UNARY_OP_HPP

#include <random>
#include <functional>
#include <ctime>

#include "math.hpp"

namespace etl {

using random_engine = std::mt19937_64;

template<typename T>
struct abs_unary_op {
    static constexpr T apply(const T& x) noexcept {
        return std::abs(x);
    }

    static std::string desc() noexcept {
        return "abs";
    }
};

template<typename T>
struct log_unary_op {
    static constexpr T apply(const T& x){
        return std::log(x);
    }

    static std::string desc() noexcept {
        return "log";
    }
};

template<typename T>
struct sqrt_unary_op {
    static constexpr T apply(const T& x){
        return std::sqrt(x);
    }

    static std::string desc() noexcept {
        return "sqrt";
    }
};

template<typename T>
struct exp_unary_op {
    static constexpr T apply(const T& x){
        return std::exp(x);
    }

    static std::string desc() noexcept {
        return "exp";
    }
};

template<typename T>
struct sign_unary_op {
    static constexpr T apply(const T& x) noexcept {
        return sign(x);
    }

    static std::string desc() noexcept {
        return "sign";
    }
};

template<typename T>
struct sigmoid_unary_op {
    static constexpr T apply(const T& x){
        return logistic_sigmoid(x);
    }

    static std::string desc() noexcept {
        return "sigmoid";
    }
};

template<typename T>
struct softplus_unary_op {
    static constexpr T apply(const T& x){
        return softplus(x);
    }

    static std::string desc() noexcept {
        return "softplus";
    }
};

template<typename T>
struct minus_unary_op {
    static constexpr T apply(const T& x) noexcept {
        return -x;
    }

    static std::string desc() noexcept {
        return "-";
    }
};

template<typename T>
struct plus_unary_op {
    static constexpr T apply(const T& x) noexcept {
        return +x;
    }

    static std::string desc() noexcept {
        return "+";
    }
};

template<typename T>
struct bernoulli_unary_op {
    static T apply(const T& x){
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        return x > generator() ? 1.0 : 0.0;
    }

    static std::string desc() noexcept {
        return "bernoulli";
    }
};

template<typename T>
struct reverse_bernoulli_unary_op {
    static T apply(const T& x){
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);
        static auto generator = std::bind(distribution, rand_engine);

        return x > generator() ? 0.0 : 1.0;
    }

    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

template<typename T>
struct uniform_noise_unary_op {
    static T apply(const T& x){
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> real_distribution(0.0, 1.0);
        static auto noise = std::bind(real_distribution, rand_engine);

        return x + noise();
    }

    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

template<typename T>
struct normal_noise_unary_op {
    static T apply(const T& x){
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);
        static auto noise = std::bind(normal_distribution, rand_engine);

        return x + noise();
    }

    static std::string desc() noexcept {
        return "normal_noise";
    }
};

template<typename T>
struct logistic_noise_unary_op {
    static T apply(const T& x){
        static random_engine rand_engine(std::time(nullptr));

        std::normal_distribution<double> noise_distribution(0.0, logistic_sigmoid(x));
        auto noise = std::bind(noise_distribution, rand_engine);

        return x + noise();
    }

    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

} //end of namespace etl

#endif
