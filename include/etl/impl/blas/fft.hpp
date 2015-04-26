//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_BLAS_FFT_HPP
#define ETL_IMPL_BLAS_FFT_HPP

#include "../../config.hpp"

#ifdef ETL_MKL_MODE
#include "mkl_dfti.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_MKL_MODE

template<typename A, typename C>
void dfft(A&&, C&&){
};

template<typename A, typename C>
void sfft(A&&, C&&){
};

#else

template<typename A, typename C>
void dfft(A&&, C&&);

template<typename A, typename C>
void sfft(A&&, C&&);

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl

#endif
