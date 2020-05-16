//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains normal generators
 */

#pragma once

#ifdef ETL_CURAND_MODE
#include "etl/impl/curand/curand.hpp"
#endif

#include <chrono> //for std::time

namespace etl {

/*!
 * \brief Generator from a normal distribution
 */
template <typename T = double>
struct normal_generator_op {
    using value_type = T; ///< The value type

    const T mean;                                      ///< The mean
    const T stddev;                                    ///< The standard deviation
    random_engine rand_engine;                         ///< The random engine
    std::normal_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable = (is_single_precision_t<T> && curand_enabled) || (is_double_precision_t<T> && curand_enabled);

    /*!
     * \brief Construct a new generator with the given mean and standard deviation
     * \param mean The mean
     * \param stddev The standard deviation
     */
    normal_generator_op(T mean, T stddev) : mean(mean), stddev(stddev), rand_engine(std::time(nullptr)), distribution(mean, stddev) {}

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

        // Generate the random numbers
        impl::curand::generate_normal(gen, t1.gpu_memory(), etl::size(y), mean, stddev);

        // Destroy the generator
        curand_call(curandDestroyGenerator(gen));

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

        // Generate the random numbers
        impl::curand::generate_normal(gen, y.gpu_memory(), etl::size(y), mean, stddev);

        // Destroy the generator
        curand_call(curandDestroyGenerator(gen));

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
    friend std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const normal_generator_op& s) {
        return os << "N(0,1)";
    }
};

/*!
 * \brief Generator from a normal distribution using a custom random engine.
 */
template <typename G, typename T = double>
struct normal_generator_g_op {
    using value_type = T; ///< The value type

    const T mean;                                      ///< The mean
    const T stddev;                                    ///< The standard deviation
    G& rand_engine;                                    ///< The random engine
    std::normal_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    static constexpr bool gpu_computable = (is_single_precision_t<T> && curand_enabled) || (is_double_precision_t<T> && curand_enabled);

    /*!
     * \brief Construct a new generator with the given mean and standard deviation
     * \param mean The mean
     * \param stddev The standard deviation
     */
    normal_generator_g_op(G& g, T mean, T stddev) : mean(mean), stddev(stddev), rand_engine(g), distribution(mean, stddev) {}

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

        // Generate the random numbers
        impl::curand::generate_normal(gen, t1.gpu_memory(), etl::size(y), mean, stddev);

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

        // Generate the random numbers
        impl::curand::generate_normal(gen, y.gpu_memory(), etl::size(y), mean, stddev);

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
    friend std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const normal_generator_g_op& s) {
        return os << "N(0,1)";
    }
};

} //end of namespace etl
