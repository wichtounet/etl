//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONFIG_HPP
#define ETL_CONFIG_HPP

namespace etl {

#ifdef ETL_BLAS_MODE

struct is_blas_enabled : std::true_type {}

#else

struct is_blas_enabled : std::false_type {}

#endif

} //end of namespace etl

#endif
