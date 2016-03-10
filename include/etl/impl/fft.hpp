//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
 * \param func The default fallback functor
 * \param args The args to be passed to the default functor
 * \return The implementation to use
 */
template<typename Functor, typename... Args>
inline fft_impl select_forced_fft_impl(Functor func, Args&&... args) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    auto forced = local_context().fft_selector.impl;

    switch (forced) {
        //MKL cannot always be used
        case fft_impl::MKL:
            if (!mkl) {
                std::cerr << "Forced selection to MKL fft implementation, but not possible for this expression" << std::endl;
                return func(std::forward<Args>(args)...);
            }

            return forced;

        //CUFFT cannot always be used
        case fft_impl::CUFFT:
            if (!cufft) {
                std::cerr << "Forced selection to CUFFT fft implementation, but not possible for this expression" << std::endl;
                return func(std::forward<Args>(args)...);
            }

            return forced;

        //In other cases, simply use the forced impl
        default:
            return forced;
    }
}

/*!
 * \brief Select a 1D FFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_default_fft1_impl(const std::size_t n) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    if (cufft) {
        if (math::is_power_of_two(n)) {
            if (n <= 64) {
                return fft_impl::STD;
            } else if (n <= 1024) {
                if (mkl) {
                    return fft_impl::MKL;
                } else {
                    return fft_impl::STD;
                }
            } else if (n <= 65536 && mkl) {
                return fft_impl::MKL;
            }

            return fft_impl::CUFFT;
        }

        if (n <= 250000 && mkl) {
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if (mkl) {
        if (math::is_power_of_two(n) && n <= 64) {
            return fft_impl::STD;
        }

        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

inline fft_impl select_fft1_impl(const std::size_t n) {
    if(local_context().fft_selector.forced){
        return select_forced_fft_impl([](std::size_t n){ return select_default_fft1_impl(n); }, n);
    }

    return select_default_fft1_impl(n);
}

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_default_fft1_many_impl(const std::size_t batch, const std::size_t n) {
    cpp_unused(batch);

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft) {
        if (n <= 250000 && mkl) {
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

inline fft_impl select_fft1_many_impl(const std::size_t batch, const std::size_t n) {
    if(local_context().fft_selector.forced){
        return select_forced_fft_impl([](std::size_t batch, std::size_t n){ return select_default_fft1_many_impl(batch, n); }, batch, n);
    }

    return select_default_fft1_many_impl(batch, n);
}

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_default_ifft1_impl(const std::size_t n) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    if (cufft) {
        if (math::is_power_of_two(n)) {
            if (n <= 1024) {
                if (mkl) {
                    return fft_impl::MKL;
                } else {
                    return fft_impl::STD;
                }
            } else if (n <= 262144 && mkl) {
                return fft_impl::MKL;
            }

            return fft_impl::CUFFT;
        }

        if (n <= 250000 && mkl) {
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

inline fft_impl select_ifft1_impl(const std::size_t n) {
    if(local_context().fft_selector.forced){
        return select_forced_fft_impl([](std::size_t n){ return select_default_ifft1_impl(n); }, n);
    }

    return select_default_ifft1_impl(n);
}

/*!
 * \brief Select a 2D FFT implementation based on the operation size
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_default_fft2_impl(const std::size_t n1, std::size_t n2) {
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    if (cufft) {
        if (math::is_power_of_two(n1) && math::is_power_of_two(n2)) {
            if (n1 * n2 < 150 * 150) {
                if (mkl) {
                    return fft_impl::MKL;
                } else {
                    return fft_impl::STD;
                }
            } else if (n1 * n2 <= 768 * 768 && mkl) {
                return fft_impl::MKL;
            }

            return fft_impl::CUFFT;
        }

        if (n1 * n2 <= 768 * 768 && mkl) {
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

inline fft_impl select_fft2_impl(const std::size_t n1, std::size_t n2) {
    if(local_context().fft_selector.forced){
        return select_forced_fft_impl([](std::size_t n1, std::size_t n2){ return select_default_fft2_impl(n1, n2); }, n1, n2);
    }

    return select_default_fft2_impl(n1, n2);
}

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_default_fft2_many_impl(const std::size_t batch, const std::size_t n1, const std::size_t n2) {
    cpp_unused(batch);

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool mkl   = is_mkl_enabled;
    constexpr const bool cufft = is_cufft_enabled;

    //Note: more testing would probably improve this selection

    if (cufft) {
        if (n1 * n2 <= 768 * 768 && mkl) {
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if (mkl) {
        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

inline fft_impl select_fft2_many_impl(const std::size_t batch, const std::size_t n1, const std::size_t n2) {
    if(local_context().fft_selector.forced){
        return select_forced_fft_impl([](std::size_t batch, std::size_t n1, std::size_t n2) {
            return select_default_fft2_many_impl(batch, n1, n2);
        }, batch, n1, n2);
    }

    return select_default_fft2_many_impl(batch, n1, n2);
}

/*!
 * \brief Functor for 1D FFT
 */
struct fft1_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft1(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft1(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 1D IFFT
 */
struct ifft1_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_ifft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 1D IFFT (real)
 */
struct ifft1_real_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_ifft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_real(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_real(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_real(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 2D FFT
 */
struct fft2_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft2(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft2(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 2D IFFT
 */
struct ifft2_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 2D IFFT (real)
 */
struct ifft2_real_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_real(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_real(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_real(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for Batched 1D FFT
 */
struct fft1_many_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const std::size_t transforms = etl::dim<0>(c);
        const std::size_t n          = etl::dim<1>(c);

        bool parallel_dispatch = select_parallel_2d(transforms, fft1_many_threshold_transforms, n, fft1_many_threshold_n);

        fft_impl impl = select_fft1_many_impl(transforms, etl::dim<1>(c));

        static cpp::default_thread_pool<> pool(threads - 1);

        if (impl == fft_impl::STD) {
            if(parallel_dispatch){
                dispatch_1d(pool, parallel_dispatch, [&](std::size_t first, std::size_t last){
                    etl::impl::standard::fft1_many(a.slice(first, last), c.slice(first, last));
                }, 0, transforms);
            } else {
                etl::impl::standard::fft1_many(std::forward<A>(a), std::forward<C>(c));
            }
        } else if (impl == fft_impl::MKL) {
            if(parallel_dispatch){
                dispatch_1d(pool, parallel_dispatch, [&](std::size_t first, std::size_t last){
                    etl::impl::blas::fft1_many(a.slice(first, last), c.slice(first, last));
                }, 0, transforms);
            } else {
                etl::impl::blas::fft1_many(std::forward<A>(a), std::forward<C>(c));
            }
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1_many(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for Batched 2D FFT
 */
struct fft2_many_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const std::size_t transforms = etl::dim<0>(c);
        const std::size_t n          = etl::size(c) / transforms;

        bool parallel_dispatch = select_parallel_2d(transforms, fft2_many_threshold_transforms, n, fft2_many_threshold_n);

        fft_impl impl = select_fft2_many_impl(etl::dim<0>(c), etl::dim<1>(c), etl::dim<2>(c));

        static cpp::default_thread_pool<> pool(threads - 1);

        if (impl == fft_impl::STD) {
            if(parallel_dispatch){
                dispatch_1d(pool, parallel_dispatch, [&](std::size_t first, std::size_t last){
                    etl::impl::standard::fft2_many(a.slice(first, last), c.slice(first, last));
                }, 0, transforms);
            } else {
                etl::impl::standard::fft2_many(std::forward<A>(a), std::forward<C>(c));
            }
        } else if (impl == fft_impl::MKL) {
            if(parallel_dispatch){
                dispatch_1d(pool, parallel_dispatch, [&](std::size_t first, std::size_t last){
                    etl::impl::blas::fft2_many(a.slice(first, last), c.slice(first, last));
                }, 0, transforms);
            } else {
                etl::impl::blas::fft2_many(std::forward<A>(a), std::forward<C>(c));
            }
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2_many(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for Batched 1D IFFT
 */
struct ifft1_many_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft1_many_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_many(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_many(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_many(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for Batched 2D IFFT
 */
struct ifft2_many_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        fft_impl impl = select_fft2_many_impl(etl::dim<0>(c), etl::dim<1>(c), etl::dim<2>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_many(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_many(std::forward<A>(a), std::forward<C>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_many(std::forward<A>(a), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 1D 'full' Convolution performed with FFT
 */
struct fft_conv1_full_impl {
    /*!
     * \brief Apply the functor
     * \param a The input matrix
     * \param b The kernel matrix
     * \param c The output sub expression
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        if(is_cufft_enabled){
            etl::impl::cufft::fft1_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if(is_mkl_enabled){
            etl::impl::blas::fft1_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else {
            etl::impl::standard::fft1_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

/*!
 * \brief Functor for 2D 'full' Convolution performed with FFT
 */
struct fft_conv2_full_impl {
    /*!
     * \brief Apply the functor
     * \param a The input matrix
     * \param b The kernel matrix
     * \param c The output sub expression
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        if(is_cufft_enabled){
            etl::impl::cufft::fft2_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else if(is_mkl_enabled){
            etl::impl::blas::fft2_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else {
            etl::impl::standard::fft2_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

} //end of namespace detail

} //end of namespace etl
