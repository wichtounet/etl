//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/threshold.hpp"

//The implementations
#include "etl/impl/std/mmul.hpp"
#include "etl/impl/std/strassen_mmul.hpp"
#include "etl/impl/blas/gemm.hpp"
#include "etl/impl/eblas/gemm.hpp"
#include "etl/impl/cublas/gemm.hpp"

namespace etl {

namespace detail {

enum class gemm_impl {
    STD,
    FAST,
    BLAS,
    CUBLAS
};

template <bool DMA, typename T>
inline cpp14_constexpr gemm_impl select_gemm_impl(const std::size_t n1, const std::size_t /*n2*/, const std::size_t n3) {
    //Only std implementation is able to handle non-dma expressions
    if (!DMA) {
        return gemm_impl::STD;
    }

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool blas   = is_cblas_enabled;
    constexpr const bool cublas = is_cublas_enabled;

    if (cublas) {
        if (n1 * n3 < gemm_cublas_min) {
            if (blas) {
                return gemm_impl::BLAS;
            }

            if (n1 * n3 < gemm_std_max) {
                return gemm_impl::STD;
            }
        }

        return gemm_impl::CUBLAS;
    } else if (blas) {
        return gemm_impl::BLAS;
    }

    //EBLAS has too much overhead for small matrices and does not handle complex numbers
    if (n1 * n3 < gemm_std_max || is_complex_t<T>::value) {
        return gemm_impl::STD;
    } else {
        return gemm_impl::FAST;
    }
}

template <bool DMA, typename T>
inline cpp14_constexpr gemm_impl select_gemv_impl(const std::size_t n1, const std::size_t n2) {
    //Only std implementation is able to handle non-dma expressions
    if (!DMA) {
        return gemm_impl::STD;
    }

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool blas   = is_cblas_enabled;
    constexpr const bool cublas = is_cublas_enabled;

    if (blas) {
        return gemm_impl::BLAS;
    }

    if (cublas) {
        if (is_complex_single_t<T>::value && n1 * n2 > 1000 * 1000) {
            return gemm_impl::CUBLAS;
        }
    }

    return gemm_impl::STD;
}

template <bool DMA, typename T>
inline cpp14_constexpr gemm_impl select_gevm_impl(const std::size_t n1, const std::size_t n2) {
    //Only std implementation is able to handle non-dma expressions
    if (!DMA) {
        return gemm_impl::STD;
    }

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool blas   = is_cblas_enabled;
    constexpr const bool cublas = is_cublas_enabled;

    if (blas) {
        return gemm_impl::BLAS;
    }

    if (cublas) {
        if (is_complex_single_t<T>::value && n1 * n2 > 1000 * 1000) {
            return gemm_impl::CUBLAS;
        }
    }

    return gemm_impl::STD;
}

template <typename A, typename B, typename C>
struct mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gemm_impl<all_dma<A, B, C>::value, value_t<A>>(etl::dim<0>(a), etl::dim<1>(a), etl::dim<1>(c));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::FAST) {
            etl::impl::eblas::gemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template <typename A, typename B, typename C>
struct vm_mul_impl {
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gevm_impl<all_dma<A, B, C>::value, value_t<A>>(etl::dim<0>(b), etl::dim<1>(b));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::vm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template <typename A, typename B, typename C>
struct mv_mul_impl {
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gemv_impl<all_dma<A, B, C>::value, value_t<A>>(etl::dim<0>(a), etl::dim<1>(a));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::mv_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template <typename A, typename B, typename C>
struct strassen_mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c) {
        etl::impl::standard::strassen_mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

} //end of namespace detail

} //end of namespace etl
