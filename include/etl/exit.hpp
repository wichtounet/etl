//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Produces exit utility when necessary
 */

#pragma once

namespace etl {

inline void exit(){
#ifdef ETL_CUDA
#ifdef ETL_GPU_POOL
    etl::gpu_memory_allocator::clear();
#endif
#endif
}

} // end of namespace etl

#define ETL_PROLOGUE etl::exit();
