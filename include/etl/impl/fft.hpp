//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/std/fft.hpp"
#include "etl/impl/blas/fft.hpp"
#include "etl/impl/cufft/fft.hpp"

namespace etl::detail {

/*!
 * \brief The precision of the operation
 */
enum class precision {
    S, ///< Single precision
    D, ///< Double precision
    C, ///< Single complex precision
    Z  ///< Double complex precision
};

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_default_fft1_impl(bool no_gpu) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !no_gpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_default_fft1_many_impl(bool no_gpu) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft && !no_gpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_default_ifft1_impl(bool no_gpu) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !no_gpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a 2D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_default_fft2_impl(bool no_gpu) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !no_gpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_default_fft2_many_impl(bool no_gpu) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft && !no_gpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param func The default fallback functor
 * \return The implementation to use
 */
inline fft_impl select_forced_fft_impl(fft_impl def) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (local_context().fft_selector.forced) {
        auto forced = local_context().fft_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case fft_impl::MKL:
                if (!mkl) {                                                                                                       //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to MKL fft implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return def;                                                                                                   //COVERAGE_EXCLUDE_LINE
                }                                                                                                                 //COVERAGE_EXCLUDE_LINE

                return forced;

            //CUFFT cannot always be used
            case fft_impl::CUFFT:
                if (!cufft || local_context().cpu) {                                                                                //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUFFT fft implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return def;                                                                                                     //COVERAGE_EXCLUDE_LINE
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
 * \brief Select a 1D FFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft1_impl() {
    return select_forced_fft_impl(select_default_fft1_impl(local_context().cpu));
}

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft1_many_impl() {
    return select_forced_fft_impl(select_default_fft1_many_impl(local_context().cpu));
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_ifft1_impl() {
    return select_forced_fft_impl(select_default_ifft1_impl(local_context().cpu));
}

/*!
 * \brief Select a 2D FFT implementation based on the operation size
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft2_impl() {
    return select_forced_fft_impl(select_default_fft2_impl(local_context().cpu));
}

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft2_many_impl() {
    return select_forced_fft_impl(select_default_fft2_many_impl(local_context().cpu));
}

#else

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_fft1_impl() {
    return (select_default_fft1_impl(false));
}

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_fft1_many_impl() {
    return (select_default_fft1_many_impl(false));
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_ifft1_impl() {
    return (select_default_ifft1_impl(false));
}

/*!
 * \brief Select a 2D FFT implementation based on the operation size
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_fft2_impl() {
    return (select_default_fft2_impl(false));
}

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
constexpr fft_impl select_fft2_many_impl() {
    return (select_default_fft2_many_impl(false));
}

#endif

/*!
 * \brief Functor for 1D FFT
 */
struct fft1_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft1_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft1(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft1(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::fft1(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Inplace 1D FFT
 */
struct inplace_fft1_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_fft1_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft1(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft1(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_fft1(c);
            }
    }
};

/*!
 * \brief Functor for 1D IFFT
 */
struct ifft1_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_ifft1_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft1(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft1(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft1(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for 1D IFFT
 */
struct inplace_ifft1_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_ifft1_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft1(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft1(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_ifft1(c);
            }
    }
};

/*!
 * \brief Functor for 1D IFFT (real)
 */
struct ifft1_real_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_ifft1_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft1_real(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft1_real(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft1_real(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for 2D FFT
 */
struct fft2_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft2_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft2(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft2(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::fft2(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for 2D FFT
 */
struct inplace_fft2_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_fft2_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft2(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft2(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_fft2(c);
            }
    }
};

/*!
 * \brief Functor for 2D IFFT
 */
struct ifft2_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft2_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft2(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft2(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft2(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for 2D IFFT
 */
struct inplace_ifft2_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_fft2_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft2(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft2(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_ifft2(c);
            }
    }
};

/*!
 * \brief Functor for 2D IFFT (real)
 */
struct ifft2_real_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft2_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft2_real(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft2_real(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft2_real(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Batched 1D FFT
 */
struct fft1_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_fft1_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft1_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft1_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::fft1_many(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Batched 1D FFT
 */
struct inplace_fft1_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        const auto impl = select_fft1_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft1_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft1_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_fft1_many(c);
            }
    }
};

/*!
 * \brief Functor for Batched 2D FFT
 */
struct fft2_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_fft2_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft2_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft2_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::fft2_many(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Batched 2D FFT
 */
struct inplace_fft2_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        const auto impl = select_fft2_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::fft2_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::fft2_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_fft2_many(c);
            }
    }
};

/*!
 * \brief Functor for Batched 1D IFFT
 */
struct ifft1_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft1_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft1_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft1_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft1_many(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Batched 1D IFFT
 */
struct inplace_ifft1_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_fft1_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft1_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft1_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_ifft1_many(c);
            }
    }
};

/*!
 * \brief Functor for Batched 2D IFFT
 */
struct ifft2_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        constexpr_select auto impl = select_fft2_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft2_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft2_many(smart_forward(a), c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::ifft2_many(smart_forward_gpu(a), c);
            }
    }
};

/*!
 * \brief Functor for Batched 2D IFFT
 */
struct inplace_ifft2_many_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename C>
    static void apply(C&& c) {
        constexpr_select auto impl = select_fft2_many_impl();

        if
            constexpr_select(impl == fft_impl::STD) {
                inc_counter("impl:std");
                etl::impl::standard::ifft2_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::MKL) {
                inc_counter("impl:mkl");
                etl::impl::blas::ifft2_many(c, c);
            }
        else if
            constexpr_select(impl == fft_impl::CUFFT) {
                inc_counter("impl:cufft");
                etl::impl::cufft::inplace_ifft2_many(c);
            }
    }
};

} //end of namespace etl::detail
