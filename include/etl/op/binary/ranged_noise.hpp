//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <ctime>

namespace etl {

/*!
 * \brief Binary operator for ranged noise generation
 *
 * This operator adds noise from N(0,1) to x. If x is 0 or the rhs
 * value, x is not modified.
 */
template <typename G, typename T, typename E>
struct ranged_noise_binary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true;  ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename L, typename R>
    static constexpr bool gpu_computable = false;

    G& rand_engine; ///< The random engine

    /*!
     * \brief Construct a new ranged_noise_binary_g_op
     */
    ranged_noise_binary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    T apply(const T& x, E value) {
        std::normal_distribution<double> normal_distribution(0.0, 1.0);
        auto noise = std::bind(normal_distribution, rand_engine);

        if (x == 0.0 || x == value) {
            return x;
        } else {
            return x + noise();
        }
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

/*!
 * \brief Binary operator for ranged noise generation
 *
 * This operator adds noise from N(0,1) to x. If x is 0 or the rhs
 * value, x is not modified.
 */
template <typename T, typename E>
struct ranged_noise_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true;  ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename L, typename R>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
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

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

} //end of namespace etl
