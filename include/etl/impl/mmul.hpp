//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//The implementations
#include "etl/impl/std/mmul.hpp"
#include "etl/impl/std/strassen_mmul.hpp"
#include "etl/impl/blas/gemm.hpp"
#include "etl/impl/vec/gemm.hpp"
#include "etl/impl/cublas/gemm.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Select an implementation of GEMM, not considering local context
 * \param n1 The left dimension of the  multiplication
 * \param n2 The inner dimension of the  multiplication
 * \param n3 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline cpp14_constexpr gemm_impl select_default_gemm_impl(const std::size_t n1, const std::size_t n2, const std::size_t n3) {
    cpp_unused(n2);

    constexpr bool DMA = all_dma<A, B, C>::value;

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool blas   = is_cblas_enabled;
    constexpr bool cublas = is_cublas_enabled;

    if (cublas && DMA) {
        if (n1 * n3 < gemm_cublas_min) {
            if (blas) {
                return gemm_impl::BLAS;
            }

            if (n1 * n3 < gemm_std_max) {
                return gemm_impl::STD;
            }
        }

        return gemm_impl::CUBLAS;
    } else if (blas && DMA) {
        return gemm_impl::BLAS;
    }

    if(vec_enabled && all_vectorizable<vector_mode, A, B, C>::value){
        return gemm_impl::VEC;
    }

    return gemm_impl::STD;
}

/*!
 * \brief Select an implementation of GEMM
 * \param n1 The left dimension of the  multiplication
 * \param n2 The inner dimension of the  multiplication
 * \param n3 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline gemm_impl select_gemm_impl(const std::size_t n1, const std::size_t n2, const std::size_t n3) {
    constexpr bool DMA = all_dma<A, B, C>::value;

    auto def = select_default_gemm_impl<A, B, C>(n1, n2, n3);

    if (local_context().gemm_selector.forced) {
        auto forced = local_context().gemm_selector.impl;

        switch (forced) {
            //CUBLAS cannot always be used
            case gemm_impl::CUBLAS:
                if (!is_cublas_enabled || !DMA) {                                                                                     //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUBLAS gemm implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return def;                                                              //COVERAGE_EXCLUDE_LINE
                }                                                                                                                     //COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case gemm_impl::BLAS:
                if (!is_cblas_enabled || !DMA) {                                                                                    //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS gemm implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return def;                                                            //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case gemm_impl::VEC:
                if (!vec_enabled || !all_vectorizable<vector_mode, A, B, C>::value) {                                               //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC gemv implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return def;                                                            //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return def;
}

/*!
 * \brief Select an implementation of GEMV, not considering local context
 * \param n1 The left dimension of the  multiplication
 * \param n2 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline cpp14_constexpr gemm_impl select_default_gemv_impl(const std::size_t n1, const std::size_t n2) {
    constexpr bool DMA = all_dma<A, B, C>::value;
    using T = value_t<A>;

    if(DMA && is_cblas_enabled){
        return gemm_impl::BLAS;
    }

    if(all_vectorizable<vector_mode, A, B, C>::value && vec_enabled){
        return gemm_impl::VEC;
    }

    if (is_cublas_enabled) {
        if (is_complex_single_t<T>::value && n1 * n2 > 1000 * 1000) {
            return gemm_impl::CUBLAS;
        }
    }

    return gemm_impl::STD;
}

/*!
 * \brief Select an implementation of GEMV
 * \param n1 The left dimension of the  multiplication
 * \param n2 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline gemm_impl select_gemv_impl(const std::size_t n1, const std::size_t n2) {
    static constexpr bool DMA = all_dma<A, B, C>::value;

    if (local_context().gemm_selector.forced) {
        auto forced = local_context().gemm_selector.impl;

        switch (forced) {
            //CUBLAS cannot always be used
            case gemm_impl::CUBLAS:
                if (!is_cublas_enabled || !DMA) {                                                                                     //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUBLAS gemv implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gemv_impl<A, B, C>(n1, n2);                                                                  //COVERAGE_EXCLUDE_LINE
                }                                                                                                                     //COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case gemm_impl::BLAS:
                if (!is_cblas_enabled || !DMA) {                                                                                    //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS gemv implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gemv_impl<A, B, C>(n1, n2);                                                                //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case gemm_impl::VEC:
                if (!vec_enabled || !all_vectorizable<vector_mode, A, B, C>::value) {                                               //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC gemv implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gemv_impl<A, B, C>(n1, n2);                                                                //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_gemv_impl<A, B, C>(n1, n2);
}

/*!
 * \brief Select an implementation of GEVM, not considering local context
 * \param n1 The left dimension of the  multiplication
 * \param n2 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline cpp14_constexpr gemm_impl select_default_gevm_impl(const std::size_t n1, const std::size_t n2) {
    constexpr bool DMA = all_dma<A, B, C>::value;
    using T = value_t<A>;

    if(DMA && is_cblas_enabled){
        return gemm_impl::BLAS;
    }

    if(all_vectorizable<vector_mode, A, B, C>::value && vec_enabled){
        return gemm_impl::VEC;
    }

    if (is_cublas_enabled) {
        if (is_complex_single_t<T>::value && n1 * n2 > 1000 * 1000) {
            return gemm_impl::CUBLAS;
        }
    }

    return gemm_impl::STD;
}

/*!
 * \brief Select an implementation of GEVM
 * \param n1 The left dimension of the  multiplication
 * \param n2 The right dimension of the  multiplication
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
inline gemm_impl select_gevm_impl(const std::size_t n1, const std::size_t n2) {
    static constexpr bool DMA = all_dma<A, B, C>::value;

    if (local_context().gemm_selector.forced) {
        auto forced = local_context().gemm_selector.impl;

        switch (forced) {
            //CUBLAS cannot always be used
            case gemm_impl::CUBLAS:
                if (!is_cublas_enabled || !DMA) {                                                                                     //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUBLAS gevm implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gevm_impl<A, B, C>(n1, n2);                                                                  //COVERAGE_EXCLUDE_LINE
                }                                                                                                                     //COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case gemm_impl::BLAS:
                if (!is_cblas_enabled || !DMA) {                                                                                    //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS gevm implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gevm_impl<A, B, C>(n1, n2);                                                                //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case gemm_impl::VEC:
                if (!vec_enabled || !all_vectorizable<vector_mode, A, B, C>::value) {                                               //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC gemv implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_gemv_impl<A, B, C>(n1, n2);                                                                //COVERAGE_EXCLUDE_LINE
                }                                                                                                                   //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_gevm_impl<A, B, C>(n1, n2);
}

/*!
 * \brief Functor for matrix-matrix multiplication
 */
struct mm_mul_impl {
    /*!
     * \brief Apply the function C = A * B
     * \param a The lhs of the multiplication
     * \param b The rhs of the multiplication
     * \param c The target of the multiplication
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gemm_impl<A, B, C>(etl::dim<0>(a), etl::dim<1>(a), etl::dim<1>(c));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::mm_mul(a, b, c);
        } else if (impl == gemm_impl::VEC) {
            etl::impl::vec::gemm(a, b, c);
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gemm(a, b, c);
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gemm(a, b, c);
        }
    }
};

/*!
 * \brief Functor for vector-matrix multiplication
 */
struct vm_mul_impl {
    /*!
     * \brief Apply the function C = A * B
     * \param a The lhs of the multiplication
     * \param b The rhs of the multiplication
     * \param c The target of the multiplication
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gevm_impl<A, B, C>(etl::dim<0>(b), etl::dim<1>(b));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::vm_mul(a, b, c);
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gevm(a, b, c);
        } else if (impl == gemm_impl::VEC) {
            etl::impl::vec::gevm(a, b, c);
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gevm(a, b, c);
        }
    }
};

/*!
 * \brief Functor for matrix-vector multiplication
 */
struct mv_mul_impl {
    /*!
     * \brief Apply the function C = A * B
     * \param a The lhs of the multiplication
     * \param b The rhs of the multiplication
     * \param c The target of the multiplication
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        gemm_impl impl = select_gemv_impl<A, B, C>(etl::dim<0>(a), etl::dim<1>(a));

        if (impl == gemm_impl::STD) {
            etl::impl::standard::mv_mul(a, b, c);
        } else if (impl == gemm_impl::BLAS) {
            etl::impl::blas::gemv(a, b, c);
        } else if (impl == gemm_impl::VEC) {
            etl::impl::vec::gemv(a, b, c);
        } else if (impl == gemm_impl::CUBLAS) {
            etl::impl::cublas::gemv(a, b, c);
        }
    }
};

/*!
 * \brief Functor for Strassen matrix-matrix multiplication
 */
struct strassen_mm_mul_impl {
    /*!
     * \brief Apply the function C = A * B
     * \param a The lhs of the multiplication
     * \param b The rhs of the multiplication
     * \param c The target of the multiplication
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        etl::impl::standard::strassen_mm_mul(a, b, c);
    }
};

} //end of namespace detail

} //end of namespace etl
