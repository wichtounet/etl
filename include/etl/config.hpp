//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONFIG_HPP
#define ETL_CONFIG_HPP

namespace etl {

#ifdef ETL_NO_TEMPORARY

struct create_temporary : std::false_type {};

#else

struct create_temporary : std::true_type {};

#endif

#ifdef ETL_BLAS_MODE

struct is_cblas_enabled : std::true_type {};

#else

struct is_cblas_enabled : std::false_type {};

#endif

#ifdef ETL_CONV1_MMUL

struct is_conv1_mmul_enabled : std::true_type {};

#else

struct is_conv1_mmul_enabled : std::false_type {};

#endif

#ifdef ETL_CONV2_MMUL

struct is_conv2_mmul_enabled : std::true_type {};

#else

struct is_conv2_mmul_enabled : std::false_type {};

#endif

} //end of namespace etl

#endif
