//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains generators
 */

#pragma once

#ifdef ETL_CURAND_MODE
#include "etl/impl/curand/curand.hpp"
#endif

#include "etl/impl/egblas/scalar_add.hpp"
#include "etl/impl/egblas/scalar_mul.hpp"

#include <chrono> //for std::time

namespace etl {

/*!
 * \brief Selector helper to get an uniform_distribution based on the type (real or int)
 * \tparam T The type of return of the distribution
 */
template <typename T>
using uniform_distribution = std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>, std::uniform_int_distribution<T>>;

/*!
 * \brief Generator from an uniform distribution
 */
template <typename T = double>
struct uniform_generator_op {
    using value_type = T; ///< The value type

    const T start;                                 ///< The start of the distribution
    const T end;                                   ///< The end of the distribution
    random_engine rand_engine;                     ///< The random engine
    uniform_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable = curand_enabled
                                           && ((is_single_precision_t<T> && impl::egblas::has_scalar_sadd && impl::egblas::has_scalar_smul)
                                               || (is_double_precision_t<T> && impl::egblas::has_scalar_dadd && impl::egblas::has_scalar_dmul));

    /*!
     * \brief Construct a new generator with the given start and end of the range
     * \param start The beginning of the range
     * \param end The end of the range
     */
    uniform_generator_op(T start, T end) : start(start), end(end), rand_engine(std::time(nullptr)), distribution(start, end) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return distribution(rand_engine);
    }

#ifdef ETL_CURAND_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename Y>
    auto gpu_compute_hint(Y& y) noexcept {
        auto t1 = etl::force_temporary_gpu_dim_only_t<T>(y);

        curandGenerator_t gen;

        // Create the generator
        curand_call(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        // Seed it with the internal random engine
        std::uniform_int_distribution<long> seed_dist;
        curand_call(curandSetPseudoRandomGeneratorSeed(gen, seed_dist(rand_engine)));

        // Generate the random numbers in [0,1]
        impl::curand::generate_uniform(gen, t1.gpu_memory(), etl::size(y));

        // mul by b-a => [0,b-a]
        auto s1 = T(end) - T(start);
        impl::egblas::scalar_mul(t1.gpu_memory(), etl::size(y), 1, s1);

        // Add a => [a,b]
        auto s2 = T(start);
        impl::egblas::scalar_add(t1.gpu_memory(), etl::size(y), 1, s2);

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
        y.ensure_gpu_allocated();

        curandGenerator_t gen;

        // Create the generator
        curand_call(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        // Seed it with the internal random engine
        std::uniform_int_distribution<long> seed_dist;
        curand_call(curandSetPseudoRandomGeneratorSeed(gen, seed_dist(rand_engine)));

        // Generate the random numbers in [0,1]
        impl::curand::generate_uniform(gen, y.gpu_memory(), etl::size(y));

        // mul by b-a => [0,b-a]
        auto s1 = T(end) - T(start);
        impl::egblas::scalar_mul(y.gpu_memory(), etl::size(y), 1, s1);

        // Add a => [a,b]
        auto s2 = T(start);
        impl::egblas::scalar_add(y.gpu_memory(), etl::size(y), 1, s2);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

#endif

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const uniform_generator_op& s) {
        cpp_unused(s);
        return os << "U(0,1)";
    }
};

/*!
 * \brief Generator from an uniform distribution using a custom random engine.
 */
template <typename G, typename T = double>
struct uniform_generator_g_op {
    using value_type = T; ///< The value type

    const T start;                                 ///< The start of the distribution
    const T end;                                   ///< The end of the distribution
    G& rand_engine;                                ///< The random engine
    uniform_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable = curand_enabled
                                           && ((is_single_precision_t<T> && impl::egblas::has_scalar_sadd && impl::egblas::has_scalar_smul)
                                               || (is_double_precision_t<T> && impl::egblas::has_scalar_dadd && impl::egblas::has_scalar_dmul));

    /*!
     * \brief Construct a new generator with the given start and end of the range
     * \param start The beginning of the range
     * \param end The end of the range
     */
    uniform_generator_g_op(G& g, T start, T end) : start(start), end(end), rand_engine(g), distribution(start, end) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return distribution(rand_engine);
    }

#ifdef ETL_CURAND_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename Y>
    auto gpu_compute_hint(Y& y) noexcept {
        auto t1 = etl::force_temporary_gpu_dim_only_t<T>(y);

        curandGenerator_t gen;

        // Create the generator
        curand_call(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        // Seed it with the internal random engine
        std::uniform_int_distribution<long> seed_dist;
        curand_call(curandSetPseudoRandomGeneratorSeed(gen, seed_dist(rand_engine)));

        // Generate the random numbers in [0,1]
        impl::curand::generate_uniform(gen, t1.gpu_memory(), etl::size(y));

        // mul by b-a => [0,b-a]
        auto s1 = T(end) - T(start);
        impl::egblas::scalar_mul(t1.gpu_memory(), etl::size(y), 1, s1);

        // Add a => [a,b]
        auto s2 = T(start);
        impl::egblas::scalar_add(t1.gpu_memory(), etl::size(y), 1, s2);

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
        y.ensure_gpu_allocated();

        curandGenerator_t gen;

        // Create the generator
        curand_call(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        // Seed it with the internal random engine
        std::uniform_int_distribution<long> seed_dist;
        curand_call(curandSetPseudoRandomGeneratorSeed(gen, seed_dist(rand_engine)));

        // Generate the random numbers in [0,1]
        impl::curand::generate_uniform(gen, y.gpu_memory(), etl::size(y));

        // mul by b-a => [0,b-a]
        auto s1 = T(end) - T(start);
        impl::egblas::scalar_mul(y.gpu_memory(), etl::size(y), 1, s1);

        // Add a => [a,b]
        auto s2 = T(start);
        impl::egblas::scalar_add(y.gpu_memory(), etl::size(y), 1, s2);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

#endif

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const uniform_generator_g_op& s) {
        cpp_unused(s);
        return os << "U(0,1)";
    }
};

} //end of namespace etl
