//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_MKL_MODE

#define disable_blas_threads()                    \
    auto etl_mkl_threads = mkl_get_max_threads(); \
    mkl_set_num_threads(1);

#define restore_blas_threads() mkl_set_num_threads(etl_mkl_threads);

#else

#define disable_blas_threads()                    \

#define restore_blas_threads()

#endif
