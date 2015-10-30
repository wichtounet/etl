//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

constexpr const std::size_t sgemm_std_max    = 75 * 75;
constexpr const std::size_t sgemm_eblas_min  = 150 * 150;
constexpr const std::size_t sgemm_cublas_min = 275 * 275;

constexpr const std::size_t dgemm_std_max    = 125 * 125;
constexpr const std::size_t dgemm_eblas_min  = 150 * 150;
constexpr const std::size_t dgemm_cublas_min = 325 * 325;

constexpr const std::size_t cgemm_std_max    = 75 * 75;
constexpr const std::size_t cgemm_cublas_min = 175 * 175;

constexpr const std::size_t zgemm_std_max    = 75 * 75;
constexpr const std::size_t zgemm_cublas_min = 180 * 180;

constexpr const std::size_t parallel_threshold = 1024;

} //end of namespace etl
