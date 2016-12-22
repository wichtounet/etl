//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDA
static_assert(false, "ETL_CUDA should never be set directly");
#endif

#ifdef ETL_VECTORIZE_FULL

//VECTORIZE_FULL enables VECTORIZE_EXPR
#ifndef ETL_VECTORIZE_EXPR
#define ETL_VECTORIZE_EXPR
#endif

//VECTORIZE_FULL enables VECTORIZE_IMPL
#ifndef ETL_VECTORIZE_IMPL
#define ETL_VECTORIZE_IMPL
#endif

#endif //ETL_VECTORIZE_FULL

//MKL mode enables BLAS mode
#ifdef ETL_MKL_MODE
#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif
#endif

// ETL_GPU enabled all GPU flags
#ifdef ETL_GPU

#ifndef ETL_CUBLAS_MODE
#define ETL_CUBLAS_MODE
#endif

#ifndef ETL_CUFFT_MODE
#define ETL_CUFFT_MODE
#endif

#ifndef ETL_CUDNN_MODE
#define ETL_CUDNN_MODE
#endif

#endif
