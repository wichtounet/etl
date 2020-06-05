//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementations of inplace matrix transposition
 */

#pragma once

//Include the implementations
#include "etl/impl/std/transpose.hpp"
#include "etl/impl/vec/transpose.hpp"
#include "etl/impl/blas/transpose.hpp"
#include "etl/impl/cublas/transpose.hpp"

#if __INTEL_MKL__ == 11 && __INTEL_MKL_MINOR__ == 2
#define SLOW_MKL
#endif

namespace etl::detail {

//TODO We should take into account parallel blas when selecting MKL transpose

/*!
 * \brief Select the default transposition implementation to use
 *
 * This does not take local context into account
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best default transpose implementation to use
 */
template <typename A, typename C>
constexpr transpose_impl select_default_transpose_impl(bool no_gpu) {
    if (cublas_enabled && all_dma<A, C> && all_floating<A, C> && !no_gpu) {
        return transpose_impl::CUBLAS;
    }

#ifdef SLOW_MKL
    // STD is always faster than MKL for out-of-place transpose
    return transpose_impl::STD;
#else
    // Condition to use MKL
    constexpr bool mkl_possible = mkl_enabled && is_dma<C> && is_floating<C>;

    if (mkl_possible) {
        return transpose_impl::MKL;
    } else {
        return transpose_impl::STD;
    }
#endif
}

/*!
 * \brief Select the default transposition implementation to use
 *
 * This does not take local context into account
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best default transpose implementation to use
 */
template <typename A, typename C>
constexpr transpose_impl select_default_oop_transpose_impl(bool no_gpu) {
    if (cublas_enabled && all_dma<A, C> && all_floating<A, C> && !no_gpu) {
        return transpose_impl::CUBLAS;
    }

    constexpr bool vec_possible = vectorize_impl && is_dma<C> && is_floating<C>;

#ifdef SLOW_MKL
    // VEC and STD is always faster than MKL for out-of-place transpose
    if (vec_possible) {
        return transpose_impl::VEC;
    } else {
        return transpose_impl::STD;
    }
#else
    // Condition to use MKL
    constexpr bool mkl_possible = mkl_enabled && is_dma<C> && is_floating<C>;

    if (mkl_possible) {
        return transpose_impl::MKL;
    } else if (vec_possible) {
        return transpose_impl::VEC;
    } else {
        return transpose_impl::STD;
    }
#endif
}

/*!
 * \brief Select the default transposition implementation to use for an inplace
 * square transposition operation.
 *
 * This does not take local context into account
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best default transpose implementation to use
 */
template <typename A, typename C>
constexpr transpose_impl select_default_in_square_transpose_impl(bool no_gpu) {
    if (cublas_enabled && all_dma<A, C> && all_floating<A, C> && !no_gpu) {
        return transpose_impl::CUBLAS;
    }

    // Condition to use MKL
    constexpr bool mkl_possible = mkl_enabled && is_dma<C> && is_floating<C>;

    if (mkl_possible) {
        return transpose_impl::MKL;
    } else {
        return transpose_impl::STD;
    }
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select the transpose implementation for an expression of type A and C
 * \tparam A The type of rhs expression
 * \tparam C The type of lhs expression
 * \return The implementation to use
 */
template <typename A, typename C>
transpose_impl select_transpose_impl(transpose_impl def) {
    if (local_context().transpose_selector.forced) {
        auto forced = local_context().transpose_selector.impl;

        switch (forced) {
            //CUBLAS cannot always be used
            case transpose_impl::CUBLAS:
                if (!cublas_enabled || !all_dma<A, C> || !all_floating<A, C> || local_context().cpu) {
                    std::cerr << "Forced selection to CUBLAS transpose implementation, but not possible for this expression" << std::endl;
                    return def;
                }

                return forced;

            //MKL cannot always be used
            case transpose_impl::MKL:
                if (!mkl_enabled || !all_dma<A, C> || !all_floating<A, C>) {
                    std::cerr << "Forced selection to MKL transpose implementation, but not possible for this expression" << std::endl;
                    return def;
                }

                return forced;

            //VEC cannot always be used
            case transpose_impl::VEC:
                if (!vectorize_impl || !all_dma<A, C> || !all_floating<A, C>) {
                    std::cerr << "Forced selection to VEC transpose implementation, but not possible for this expression" << std::endl;
                    return def;
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return def;
}

/*!
 * \brief Select the transposition implementation to use
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename A, typename C>
transpose_impl select_normal_transpose_impl() {
    return select_transpose_impl<A, C>(select_default_transpose_impl<A, C>(local_context().cpu));
}

/*!
 * \brief Select the transposition implementation to use
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename A, typename C>
transpose_impl select_oop_transpose_impl() {
    return select_transpose_impl<A, C>(select_default_oop_transpose_impl<A, C>(local_context().cpu));
}

/*!
 * \brief Select the transposition implementation to use for an inplace
 * square transposition operation.
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename C>
transpose_impl select_in_square_transpose_impl() {
    return select_transpose_impl<C, C>(select_default_in_square_transpose_impl<C, C>(local_context().cpu));
}

#else

/*!
 * \brief Select the transposition implementation to use
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename A, typename C>
constexpr transpose_impl select_normal_transpose_impl() {
    return select_default_transpose_impl<A, C>(false);
}

/*!
 * \brief Select the transposition implementation to use
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename A, typename C>
constexpr transpose_impl select_oop_transpose_impl() {
    return select_default_oop_transpose_impl<A, C>(false);
}

/*!
 * \brief Select the transposition implementation to use for an inplace
 * square transposition operation.
 *
 * \tparam A The type of input
 * \tparam C The type of output
 *
 * \return The best transpose implementation to use
 */
template <typename C>
constexpr transpose_impl select_in_square_transpose_impl() {
    return select_default_in_square_transpose_impl<C, C>(false);
}

#endif

/*!
 * \brief Functor for inplace square matrix transposition
 */
struct inplace_square_transpose {
    /*!
     * \brief Tranpose c inplace
     * \param c The target matrix
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select const auto impl = select_in_square_transpose_impl<C>();

        if
            constexpr_select(impl == transpose_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::inplace_square_transpose(c);
            }
        else if
            constexpr_select(impl == transpose_impl::CUBLAS) {
                inc_counter("impl:cublas");
                etl::impl::cublas::inplace_square_transpose(c);
            }
        else if
            constexpr_select(impl == transpose_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::inplace_square_transpose(c);
            }
        else {
            cpp_unreachable("Invalid transpose_impl selection");
        }
    }
};

/*!
 * \brief Functor for inplace rectangular matrix transposition
 */
struct inplace_rectangular_transpose {
    /*!
     * \brief Tranpose c inplace
     * \param c The target matrix
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select const auto impl = select_normal_transpose_impl<C, C>();

        if
            constexpr_select(impl == transpose_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::inplace_rectangular_transpose(c);
            }
        else if
            constexpr_select(impl == transpose_impl::CUBLAS) {
                inc_counter("impl:cublas");
                etl::impl::cublas::inplace_rectangular_transpose(c);
            }
        else if
            constexpr_select(impl == transpose_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::inplace_rectangular_transpose(c);
            }
        else {
            cpp_unreachable("Invalid transpose_impl selection");
        }
    }
};

/*!
 * \brief Functor for general matrix transposition
 */
struct transpose {
    /*!
     * \brief Tranpose a and store the results in c
     * \param a The source matrix
     * \param c The target matrix
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select const auto impl = select_oop_transpose_impl<A, C>();

        if
            constexpr_select(impl == transpose_impl::CUBLAS) {
                c.ensure_gpu_allocated();

                decltype(auto) aa = smart_forward_gpu(a);

                // Detect inplace (some implementations do not support inplace if not told explicitely)
                if (aa.gpu_memory() && aa.gpu_memory() == c.gpu_memory()) {
                    if (is_square(c)) {
                        inplace_square_transpose::apply(c);
                    } else {
                        inplace_rectangular_transpose::apply(c);
                    }

                    return;
                }

                inc_counter("impl:cublas");
                etl::impl::cublas::transpose(aa, c);
            }
        else {
            decltype(auto) aa = smart_forward(a);

            // Detect inplace (some implementations do not support inplace if not told explicitely)
            if (aa.memory_start() == c.memory_start()) {
                if (is_square(c)) {
                    inplace_square_transpose::apply(c);
                } else {
                    inplace_rectangular_transpose::apply(c);
                }

                return;
            }

            if
                constexpr_select(impl == transpose_impl::MKL) {
                    inc_counter("impl:mkl");
                    etl::impl::blas::transpose(aa, c);
                }
            else if
                constexpr_select(impl == transpose_impl::VEC) {
                    inc_counter("impl:vec");
                    etl::impl::vec::transpose(aa, c);
                }
            else if
                constexpr_select(impl == transpose_impl::STD) {
                    inc_counter("impl:std");
                    etl::impl::standard::transpose(aa, c);
                }
            else {
                cpp_unreachable("Invalid transpose_impl selection");
            }
        }
    }
};

} //end of namespace etl::detail
