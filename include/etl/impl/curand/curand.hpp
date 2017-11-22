//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace curand {

#define curand_call(call)                                                             \
    {                                                                                 \
        auto status = call;                                                           \
        if (status != CURAND_STATUS_SUCCESS) {                                        \
            std::cerr << "CURAND error: " << status << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;         \
        }                                                                             \
    }

void generate_normal(curandGenerator_t generator, float* gpu_memory, size_t n, float mean, float stddev){
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

void generate_normal(curandGenerator_t generator, double* gpu_memory, size_t n, double mean, double stddev){
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

void generate_uniform(curandGenerator_t generator, float* gpu_memory, size_t n){
    curand_call(curandGenerateUniform(generator, gpu_memory, n));
}

void generate_uniform(curandGenerator_t generator, double* gpu_memory, size_t n){
    curand_call(curandGenerateUniformDouble(generator, gpu_memory, n));
}

} //end of namespace curand

} //end of namespace impl

} //end of namespace etl
