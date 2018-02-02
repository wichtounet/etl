//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Utility functions for curand
 */

#pragma once

#include "curand.h"

namespace etl::impl::curand {

#define curand_call(call)                                                             \
    {                                                                                 \
        auto status = call;                                                           \
        if (status != CURAND_STATUS_SUCCESS) {                                        \
            std::cerr << "CURAND error: " << status << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;         \
        }                                                                             \
    }

/*!
 * \brief Generate a normal distribution with the given mean and
 * standard deviation with CURAND, in single-precision.
 *
 * \param generator The configured CURAND generator
 * \param gpu_memory Pointer to the GPU memory to fill
 * \param n The number of elements to set
 * \param mean The mean of the distribution to generate.
 * \param stddev The standard deviation of the distribution to generate.
 */
inline void generate_normal(curandGenerator_t generator, float* gpu_memory, size_t n, float mean, float stddev){
    // Note: CURAND is dumb, cannot generate odd sequences...

    if(n % 2 == 0){
        curand_call(curandGenerateNormal(generator, gpu_memory, n, mean, stddev));
    } else {
        // Generate the first n - 1 numbers
        curand_call(curandGenerateNormal(generator, gpu_memory, n - 1, mean, stddev));

        // Generate the last two numbers
        curand_call(curandGenerateNormal(generator, gpu_memory + (n - 3), 2, mean, stddev));
    }
}

/*!
 * \brief Generate a normal distribution with the given mean and
 * standard deviation with CURAND, in double-precision.
 *
 * \param generator The configured CURAND generator
 * \param gpu_memory Pointer to the GPU memory to fill
 * \param n The number of elements to set
 * \param mean The mean of the distribution to generate.
 * \param stddev The standard deviation of the distribution to generate.
 */
inline void generate_normal(curandGenerator_t generator, double* gpu_memory, size_t n, double mean, double stddev){
    // Note: CURAND is dumb, cannot generate odd sequences...

    if(n % 2 == 0){
        curand_call(curandGenerateNormalDouble(generator, gpu_memory, n, mean, stddev));
    } else {
        // Generate the first n - 1 numbers
        curand_call(curandGenerateNormalDouble(generator, gpu_memory, n - 1, mean, stddev));

        // Generate the last two numbers
        curand_call(curandGenerateNormalDouble(generator, gpu_memory + (n - 3), 2, mean, stddev));
    }
}

/*!
 * \brief Generate a uniform distribution between 0 and 1, in single
 * precision.
 *
 * \param generator The configured CURAND generator
 * \param gpu_memory Pointer to the GPU memory to fill
 * \param n The number of elements to set
 */
inline void generate_uniform(curandGenerator_t generator, float* gpu_memory, size_t n){
    curand_call(curandGenerateUniform(generator, gpu_memory, n));
}

/*!
 * \brief Generate a uniform distribution between 0 and 1, in single
 * precision.
 *
 * \param generator The configured CURAND generator
 * \param gpu_memory Pointer to the GPU memory to fill
 * \param n The number of elements to set
 */
inline void generate_uniform(curandGenerator_t generator, double* gpu_memory, size_t n){
    curand_call(curandGenerateUniformDouble(generator, gpu_memory, n));
}

} //end of namespace etl::impl::curand
