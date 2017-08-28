//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/std/fft.hpp"
#include "etl/impl/blas/fft.hpp"
#include "etl/impl/cufft/fft.hpp"

namespace etl {

namespace detail {

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
 * \param func The default fallback functor
 * \return The implementation to use
 */
template <typename Functor>
inline fft_impl select_forced_fft_impl(Functor func) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    auto forced = local_context().fft_selector.impl;

    switch (forced) {
        //MKL cannot always be used
        case fft_impl::MKL:
            if (!mkl) {                                                                                                       //COVERAGE_EXCLUDE_LINE
                std::cerr << "Forced selection to MKL fft implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                return func();                                                                                         //COVERAGE_EXCLUDE_LINE
            }                                                                                                                 //COVERAGE_EXCLUDE_LINE

            return forced;

        //CUFFT cannot always be used
        case fft_impl::CUFFT:
            if (!cufft || local_context().cpu) {                                                                                                       //COVERAGE_EXCLUDE_LINE
                std::cerr << "Forced selection to CUFFT fft implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                return func();                                                                                           //COVERAGE_EXCLUDE_LINE
            }                                                                                                                   //COVERAGE_EXCLUDE_LINE

            return forced;

        //In other cases, simply use the forced impl
        default:
            return forced;
    }
}

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_default_fft1_impl() {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !local_context().cpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft1_impl() {
    if (local_context().fft_selector.forced) {
        return select_forced_fft_impl([]() { return select_default_fft1_impl(); });
    }

    return select_default_fft1_impl();
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
inline fft_impl select_default_fft1_many_impl() {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft && !local_context().cpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft1_many_impl() {
    if (local_context().fft_selector.forced) {
        return select_forced_fft_impl([]() { return select_default_fft1_many_impl(); });
    }

    return select_default_fft1_many_impl();
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 *
 * This does not consider the local context configuration.
 *
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_default_ifft1_impl() {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !local_context().cpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline fft_impl select_ifft1_impl() {
    if (local_context().fft_selector.forced) {
        return select_forced_fft_impl([]() { return select_default_ifft1_impl(); });
    }

    return select_default_ifft1_impl();
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
inline fft_impl select_default_fft2_impl() {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    if (cufft && !local_context().cpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a 2D FFT implementation based on the operation size
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft2_impl() {
    if (local_context().fft_selector.forced) {
        return select_forced_fft_impl([]() { return select_default_fft2_impl(); });
    }

    return select_default_fft2_impl();
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
inline fft_impl select_default_fft2_many_impl() {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr bool mkl   = mkl_enabled;
    constexpr bool cufft = cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft && !local_context().cpu) {
        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline fft_impl select_fft2_many_impl() {
    if (local_context().fft_selector.forced) {
        return select_forced_fft_impl([]() {
            return select_default_fft2_many_impl();
        });
    }

    return select_default_fft2_many_impl();
}

/*!
 * \brief Functor for 1D FFT
 */
struct fft1_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft1_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft1(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft1(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_ifft1_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_ifft1_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_real(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_real(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_real(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft2(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft2(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_real(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_real(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_real(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_fft1_many_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft1_many(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft1_many(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1_many(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_fft2_many_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft2_many(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft2_many(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2_many(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft1_many_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_many(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_many(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_many(a, c);
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
    template<typename A>
    static constexpr bool gpu_computable = cufft_enabled;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_many_impl();

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_many(a, c);
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_many(a, c);
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_many(a, c);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
