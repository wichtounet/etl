//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains generators for inverted_dropout mask
 */

#pragma once

#include <chrono> //for std::time

namespace etl {

/*!
 * \brief Generator from an uniform distribution
 */
template <typename T = double>
struct inverted_dropout_mask_generator_op {
    using value_type = T; ///< The value type

    const T probability;                           ///< The dropout probability
    random_engine rand_engine;                     ///< The random engine
    dropout_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable =
        (is_single_precision_t<T> && impl::egblas::has_sinv_dropout_seed) || (is_double_precision_t<T> && impl::egblas::has_dinv_dropout_seed);

    /*!
     * \brief Construct a new generator with the given start and end of the range
     */
    inverted_dropout_mask_generator_op(T probability) : probability(probability), rand_engine(std::time(nullptr)), distribution(T(0), T(1)) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        if (distribution(rand_engine) < probability) {
            return T(0);
        } else {
            return T(1) / (T(1) - probability);
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
        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = force_temporary_gpu_dim_only(y);

        T alpha(1.0);
        impl::egblas::inv_dropout_seed(etl::size(y), probability, alpha, t1.gpu_memory(), 1, seed_dist(rand_engine));

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
        std::uniform_int_distribution<long> seed_dist;

        T alpha(1.0);
        impl::egblas::inv_dropout_seed(etl::size(y), probability, alpha, y.gpu_memory(), 1, seed_dist(rand_engine));

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
    friend std::ostream& operator<<(std::ostream& os, const inverted_dropout_mask_generator_op& s) {
        return os << "inverted_dropout(p=" << s.probability << ")";
    }
};

/*!
 * \brief Generator from an uniform distribution using a custom random engine.
 */
template <typename G, typename T = double>
struct inverted_dropout_mask_generator_g_op {
    using value_type = T; ///< The value type

    const T probability;                           ///< The dropout probability
    G& rand_engine;                                ///< The random engine
    dropout_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable =
        (is_single_precision_t<T> && impl::egblas::has_sinv_dropout_seed) || (is_double_precision_t<T> && impl::egblas::has_dinv_dropout_seed);

    /*!
     * \brief Construct a new generator with the given start and end of the range
     * \param start The beginning of the range
     * \param end The end of the range
     */
    inverted_dropout_mask_generator_g_op(G& g, T probability) : probability(probability), rand_engine(g), distribution(T(0), T(1)) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        if (distribution(rand_engine) < probability) {
            return T(0);
        } else {
            return T(1) / (T(1) - probability);
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
        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = force_temporary_gpu_dim_only(y);

        T alpha(1.0);
        impl::egblas::inv_dropout_seed(etl::size(y), probability, alpha, t1.gpu_memory(), 1, seed_dist(rand_engine));

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
        std::uniform_int_distribution<long> seed_dist;

        T alpha(1.0);
        impl::egblas::inv_dropout_seed(etl::size(y), probability, alpha, y.gpu_memory(), 1, seed_dist(rand_engine));

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
    friend std::ostream& operator<<(std::ostream& os, const inverted_dropout_mask_generator_g_op& s) {
        return os << "inverted_dropout(p=" << s.probability << ")";
    }
};

} //end of namespace etl
