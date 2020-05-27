//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains generators for state dropout mask
 */

#pragma once

#include <chrono> //for std::time

#include "etl/impl/egblas/dropout.hpp"

namespace etl {

/*!
 * \brief Generator from an uniform distribution
 */
template <typename T = double>
struct state_dropout_mask_generator_op {
    using value_type = T; ///< The value type

    const T probability; ///< The dropout probability
    std::shared_ptr<void*> states;
    random_engine rand_engine;                     ///< The random engine
    dropout_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable =
        impl::egblas::has_dropout_prepare && impl::egblas::has_dropout_release
        && ((is_single_precision_t<T> && impl::egblas::has_sdropout_states) || (is_double_precision_t<T> && impl::egblas::has_ddropout_states));

    /*!
     * \brief Construct a new generator with the given start and end of the range
     */
    state_dropout_mask_generator_op(T probability) : probability(probability), rand_engine(std::time(nullptr)), distribution(T(0), T(1)) {
        if constexpr (impl::egblas::has_dropout_prepare) {
            states  = std::make_shared<void*>();
            *states = impl::egblas::dropout_prepare();
        }
    }

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        if (distribution(rand_engine) < probability) {
            return T(0);
        } else {
            return T(1);
        }
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename Y>
    auto gpu_compute_hint(Y& y) noexcept {
        decltype(auto) t1 = force_temporary_gpu_dim_only(y);

        impl::egblas::dropout_states(etl::size(y), probability, T(1), t1.gpu_memory(), 1, *states);

        return t1;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename Y>
    Y& gpu_compute(Y& y) noexcept {
        impl::egblas::dropout_states(etl::size(y), probability, T(1), y.gpu_memory(), 1, *states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const state_dropout_mask_generator_op& s) {
        return os << "dropout(p=" << s.probability << ")";
    }
};

/*!
 * \brief Generator from an uniform distribution using a custom random engine.
 */
template <typename G, typename T = double>
struct state_dropout_mask_generator_g_op {
    using value_type = T; ///< The value type

    const T probability; ///< The dropout probability
    G& rand_engine;      ///< The random engine
    std::shared_ptr<void*> states;
    dropout_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable =
        impl::egblas::has_dropout_prepare_seed && impl::egblas::has_dropout_release
        && ((is_single_precision_t<T> && impl::egblas::has_sdropout_states) || (is_double_precision_t<T> && impl::egblas::has_ddropout_states));

    /*!
     * \brief Construct a new generator with the given start and end of the range
     * \param start The beginning of the range
     * \param end The end of the range
     */
    state_dropout_mask_generator_g_op(G& g, T probability) : probability(probability), rand_engine(g), distribution(T(0), T(1)) {
        if constexpr (impl::egblas::has_dropout_prepare) {
            states = std::make_shared<void*>();

            std::uniform_int_distribution<long> seed_dist;
            *states = impl::egblas::dropout_prepare_seed(seed_dist(rand_engine));
        }
    }

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        if (distribution(rand_engine) < probability) {
            return T(0);
        } else {
            return T(1);
        }
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename Y>
    auto gpu_compute_hint(Y& y) noexcept {
        decltype(auto) t1 = force_temporary_gpu_dim_only(y);

        impl::egblas::dropout_states(etl::size(y), probability, T(1), t1.gpu_memory(), 1, states);

        return t1;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename Y>
    Y& gpu_compute(Y& y) noexcept {
        impl::egblas::dropout_states(etl::size(y), probability, T(1), y.gpu_memory(), 1, states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const state_dropout_mask_generator_g_op& s) {
        return os << "dropout(p=" << s.probability << ")";
    }
};

} //end of namespace etl
