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
 * \brief The different FFT implementations
 */
enum class fft_impl {
    STD,  ///< The standard implementation
    MKL,  ///< The Intel MKL implementation
    CUFFT ///< The NVidia CuFFT implementation
};

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
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_fft1_impl(const std::size_t n) {
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

/*!
 * \brief Select a Many-1D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_fft1_many_impl(const std::size_t batch, const std::size_t n) {
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

/*!
 * \brief Select a 1D IFFT implementation based on the operation size
 * \param n The size of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_ifft1_impl(const std::size_t n) {
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


/*!
 * \brief Select a 2D FFT implementation based on the operation size
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_fft2_impl(const std::size_t n1, std::size_t n2) {
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

/*!
 * \brief Select a Many-2D FFT implementation based on the operation size
 * \param batch The number of operations
 * \param n1 The first dimension of the operation
 * \param n2 The second dimension of the operation
 * \return The implementation to use
 */
inline cpp14_constexpr fft_impl select_fft2_many_impl(const std::size_t batch, const std::size_t n1, const std::size_t n2) {
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

template <typename A, typename C>
struct fft1_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft1_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_ifft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft1_real_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_ifft1_impl(etl::size(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_real(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_real(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_real(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct fft2_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft2(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft2(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft2_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft2_real_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft2_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_real(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_real(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_real(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct fft1_many_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        const std::size_t transforms = etl::dim<0>(c);
        const std::size_t n          = etl::dim<1>(c);

        bool parallel_dispatch = select_parallel_2d(transforms, fft1_many_threshold_transforms, n, fft1_many_threshold_n);

        fft_impl impl = select_fft1_many_impl(transforms, etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            if(parallel_dispatch){
                dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                    //TODO Ideally here, we should use fft1_many on a slice
                    for(std::size_t k = first; k < last; ++k){
                        etl::impl::standard::fft1(a(k), c(k));
                    }
                }, 0, transforms);
            } else {
                etl::impl::standard::fft1_many(std::forward<AA>(a), std::forward<CC>(c));
            }
        } else if (impl == fft_impl::MKL) {
            if(parallel_dispatch){
                dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                    //TODO Ideally here, we should use fft1_many on a slice
                    for(std::size_t k = first; k < last; ++k){
                        etl::impl::blas::fft1(a(k), c(k));
                    }
                }, 0, transforms);
            } else {
                etl::impl::blas::fft1_many(std::forward<AA>(a), std::forward<CC>(c));
            }
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft1_many(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct fft2_many_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft2_many_impl(etl::dim<0>(c), etl::dim<1>(c), etl::dim<2>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::fft2_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::fft2_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::fft2_many(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft1_many_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft1_many_impl(etl::dim<0>(c), etl::dim<1>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft1_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft1_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft1_many(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename C>
struct ifft2_many_impl {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        fft_impl impl = select_fft2_many_impl(etl::dim<0>(c), etl::dim<1>(c), etl::dim<2>(c));

        if (impl == fft_impl::STD) {
            etl::impl::standard::ifft2_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::MKL) {
            etl::impl::blas::ifft2_many(std::forward<AA>(a), std::forward<CC>(c));
        } else if (impl == fft_impl::CUFFT) {
            etl::impl::cufft::ifft2_many(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template <typename A, typename B, typename C, typename Enable = void>
struct fft_conv1_full_impl {
    template <typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c) {
        if(is_cufft_enabled){
            etl::impl::cufft::fft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        } else if(is_mkl_enabled){
            etl::impl::blas::fft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        } else {
            etl::impl::standard::fft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        }
    }
};

template <typename A, typename B, typename C, typename Enable = void>
struct fft_conv2_full_impl {
    template <typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c) {
        if(is_cufft_enabled){
            etl::impl::cufft::fft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        } else if(is_mkl_enabled){
            etl::impl::blas::fft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        } else {
            etl::impl::standard::fft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
        }
    }
};

} //end of namespace detail

} //end of namespace etl
