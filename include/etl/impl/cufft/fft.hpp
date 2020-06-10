//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUFFT_MODE
#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cufft/cufft.hpp"

#ifdef ETL_CUBLAS_MODE
#include "etl/impl/cublas/cublas.hpp"
#endif

#ifndef ETL_CUBLAS_MODE
#ifndef ETL_NO_WARNINGS
#warning "Using CUBLAS with CUFFT improves the performance"
#endif
#endif

#endif

namespace etl::impl::cufft {

/*!
 * \brief Traits indicating if 1D FFT with CUFFT is
 * possible for the given configuration.
 *
 * \param A The type of the input matrix
 * \param B The type of the output matrix
 */
template <typename A, typename B>
constexpr bool fft1_possible = cufft_enabled
                               && ((is_deep_single_precision<A> && is_deep_single_precision<B>) || (is_deep_double_precision<A> && is_deep_double_precision<B>))
                               && all_dma<A, B>;

/*!
 * \brief Traits indicating if 2D FFT with CUFFT is
 * possible for the given configuration.
 *
 * \param A The type of the input matrix
 * \param B The type of the output matrix
 */
template <typename A, typename B>
constexpr bool fft2_possible = cufft_enabled
                               && ((is_deep_single_precision<A> && is_deep_single_precision<B>) || (is_deep_double_precision<A> && is_deep_double_precision<B>))
                               && all_row_major<A, B>&& all_dma<A, B>;

/*!
 * \brief Traits indicating if 1D Convolution with CUFFT is
 * possible for the given configuration.
 *
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 * \param C The type of the output matrix
 */
template <typename I, typename K, typename C>
constexpr bool conv1_possible = cufft_enabled&& all_homogeneous<I, K, C>&& all_floating<I, K, C>&& all_dma<I, K, C>;

/*!
 * \brief Traits indicating if 2D Convolution with CUFFT is
 * possible for the given configuration.
 *
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 * \param C The type of the output matrix
 */
template <typename I, typename K, typename C>
constexpr bool conv2_possible = cufft_enabled&& all_homogeneous<I, K, C>&& all_floating<I, K, C>&& all_row_major<I, K, C>&& all_dma<I, K, C>;

#ifdef ETL_CUFFT_MODE

namespace detail {

/*!
 * \brief Wrapper for cufftExecC2C, for single precision
 * \param plan The CUFFT plan
 * \param idata The input data
 * \param odata The output data
 * \param direction The direction of the transform
 */
inline void cufft_exec_c2c(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) {
    cufft_check(cufftExecC2C(plan, idata, odata, direction));
}

/*!
 * \brief Wrapper for cufftExecC2C, for double precision
 * \param plan The CUFFT plan
 * \param idata The input data
 * \param odata The output data
 * \param direction The direction of the transform
 */
inline void cufft_exec_c2c(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) {
    cufft_check(cufftExecZ2Z(plan, idata, odata, direction));
}

/*!
 * \brief Inplace 1D FFT kernel
 * \param a The matrix
 * \param n The size of the transform
 */
template <typename T>
void inplace_fft1_kernel(T&& a, size_t n) {
    decltype(auto) handle = start_cufft();

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlan1d(&handle.get(), n, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, 1));
    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace batched single-precision 1D FFT kernel
 * \param a The matrix
 * \param batch The number of transform
 * \param n The size of the transform
 */
template <typename T>
void inplace_fft1_many_kernel(T&& a, size_t batch, size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlanMany(&handle.get(), 1, dims, nullptr, 1, n, nullptr, 1, n, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, batch));

    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace single-precision 1D Inverse FFT kernel
 * \param a The matrix
 * \param batch The number of transform
 * \param n The size of the transform
 */
template <typename T>
void inplace_ifft1_kernel(T&& a, size_t n) {
    decltype(auto) handle = start_cufft();

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlan1d(&handle.get(), n, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, 1));
    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace batched single-precision 1D Inverse FFT kernel
 * \param a The matrix
 * \param batch The number of transform
 * \param n The size of the transform
 */
template <typename T>
void inplace_ifft1_many_kernel(T&& a, size_t batch, size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlanMany(&handle.get(), 1, dims, nullptr, 1, n, nullptr, 1, n, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, batch));

    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace single-precision 2D FFT kernel
 * \param a The matrix
 * \param d1 The first dimension of the transform
 * \param d2 The second dimension of the transform
 */
template <typename T>
inline void inplace_fft2_kernel(T&& a, size_t d1, size_t d2) {
    decltype(auto) handle = start_cufft();

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlan2d(&handle.get(), d1, d2, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z));
    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace batched double-precision 2D FFT kernel
 * \param a The matrix
 * \param batch The number of transforms
 * \param d1 The first dimension of the transform
 * \param d2 The second dimension of the transform
 */
template <typename T>
void inplace_fft2_many_kernel(T&& a, size_t batch, size_t d1, size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, batch));

    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace single-precision 2D Inverse FFT kernel
 * \param a The matrix
 * \param d1 The first dimension of the transform
 * \param d2 The second dimension of the transform
 */
template <typename T>
void inplace_ifft2_kernel(T&& a, size_t d1, size_t d2) {
    decltype(auto) handle = start_cufft();

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlan2d(&handle.get(), d1, d2, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z));
    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);

    a.invalidate_cpu();
}

/*!
 * \brief Inplace batched double-precision 2D Inverse FFT kernel
 * \param a The matrix
 * \param batch The number of transforms
 * \param d1 The first dimension of the transform
 * \param d2 The second dimension of the transform
 */
template <typename T>
void inplace_ifft2_many_kernel(T&& a, size_t batch, size_t d1, size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    a.ensure_gpu_up_to_date();

    cufft_check(cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, is_complex_single_precision<T> ? CUFFT_C2C : CUFFT_Z2Z, batch));

    cufft_exec_c2c(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);

    a.invalidate_cpu();
}

/*!
 * \brief Compute the 2D full convolution of a with the kernel from
 * b and store the result in c.
 *
 * c = beta * c + conv_full(a, b)
 *
 * \param a The input
 * \param m1 The first dimension of the input
 * \param m2 The second dimension of the input
 * \param b The kernel
 * \param n1 The first dimension of the kernel
 * \param n2 The second dimension of the kernel
 * \param c The output
 * \param beta Scaling factor for the output
 */
template <typename T>
void conv2_full_kernel(const T* a, size_t m1, size_t m2, const T* b, size_t n1, size_t n2, T* c, T beta) {
    const size_t s1   = m1 + n1 - 1;
    const size_t s2   = m2 + n2 - 1;
    const size_t size = s1 * s2;

    decltype(auto) handle = start_cufft();

    dyn_vector<etl::complex<T>> a_padded(size);
    dyn_vector<etl::complex<T>> b_padded(size);

    for (size_t i = 0; i < m1; ++i) {
        direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
    }

    for (size_t i = 0; i < n1; ++i) {
        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
    }

    // FFT of the two padded matrices

    inplace_fft2_kernel(a_padded, s1, s2);
    inplace_fft2_kernel(b_padded, s1, s2);

    // Element-wise matrix multiplication

    a_padded *= b_padded;

    // Inverse FFT

    inplace_ifft2_kernel(a_padded, s1, s2);

    // Scale back

    a_padded.ensure_cpu_up_to_date();

    if (beta == T(0.0)) {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a_padded[i].real * (T(1.0) / size);
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            c[i] = beta * c[i] + a_padded[i].real * (T(1.0) / size);
        }
    }
}

/*!
 * \brief Scale back the results after the inverse FFT
 * \param c The result of the inverse FFT
 * \param factor The scaling factor
 */
template <typename C>
void scale_back(C&& c, float factor) {
    if constexpr (is_complex_single_precision<C>) {
#ifdef ETL_CUBLAS_MODE
        decltype(auto) handle = impl::cublas::start_cublas();

        cuComplex alpha = make_cuComplex(factor, 0.0);
        cublas_check(cublasCscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuComplex*>(c.gpu_memory()), 1));

        //The CPU memory is not up-to-date
        c.invalidate_cpu();
#else
        //Copy from GPU and scale on CPU
        c.ensure_cpu_up_to_date();
        c *= factor;

        //The GPU memory is not up-to-date
        c.invalidate_gpu();
#endif
    } else if constexpr (is_complex_double_precision<C>) {
#ifdef ETL_CUBLAS_MODE
        decltype(auto) handle = impl::cublas::start_cublas();

        cuDoubleComplex alpha = make_cuDoubleComplex(factor, 0.0);
        cublas_check(cublasZscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1));

        //The CPU memory is not up-to-date
        c.invalidate_cpu();
#else
        //Copy from GPU and scale on CPU
        c.ensure_cpu_up_to_date();
        c *= factor;

        //The GPU memory is not up-to-date
        c.invalidate_gpu();
#endif
    }
}

/*!
 * \brief Scale back the results after the inverse FFT
 * \param c The result of the inverse FFT
 */
template <typename C>
void scale_back(C&& c) {
    scale_back(std::forward<C>(c), 1.0 / etl::size(c));
}

/*!
 * \brief Scale back the results after the inverse FFT, keeping only the real part of the results
 * \param a The result of the inverse FFT
 * \param c The final results
 */
template <typename A, typename C>
void scale_back_real(A&& a, C&& c) {
    if constexpr (is_complex_single_precision<A>) {
#ifdef ETL_CUBLAS_MODE
        c.ensure_gpu_allocated();

        decltype(auto) handle = impl::cublas::start_cublas();

        //Copy the real part of a to c
        cublas_check(cublasScopy(handle.get(), etl::size(c), reinterpret_cast<float*>(a.gpu_memory()), 2, reinterpret_cast<float*>(c.gpu_memory()), 1));

        //Scale c
        float alpha = 1.0 / etl::size(c);
        cublas_check(cublasSscal(handle.get(), etl::size(c), &alpha, c.gpu_memory(), 1));

        //The CPU memory is not up-to-date
        c.validate_gpu();
        c.invalidate_cpu();
#else
        auto tmp = allocate<std::complex<float>>(etl::size(a));

        cuda_check(cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < etl::size(a); ++i) {
            c[i] = tmp[i].real() / etl::size(a);
        }

        //The GPU memory is not up-to-date
        c.validate_cpu();
        c.invalidate_gpu();
#endif
    } else if constexpr (is_complex_double_precision<A>) {
#ifdef ETL_CUBLAS_MODE
        c.ensure_gpu_allocated();

        decltype(auto) handle = impl::cublas::start_cublas();

        //Copy the real part of a to c
        cublas_check(cublasDcopy(handle.get(), etl::size(c), reinterpret_cast<double*>(a.gpu_memory()), 2, reinterpret_cast<double*>(c.gpu_memory()), 1));

        //Scale c
        double alpha = 1.0 / etl::size(c);
        cublas_check(cublasDscal(handle.get(), etl::size(c), &alpha, c.gpu_memory(), 1));

        //The CPU memory is not up-to-date
        c.validate_gpu();
        c.invalidate_cpu();
#else
        auto tmp = allocate<std::complex<double>>(etl::size(a));

        cuda_check(cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < etl::size(a); ++i) {
            c[i] = tmp[i].real() / etl::size(a);
        }

        //The GPU memory is not up-to-date
        c.validate_cpu();
        c.invalidate_gpu();
#endif
    }
}

} //End of namespace detail

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft1(A&& a, C&& c) {
    if constexpr (is_complex<A>) {
        a.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        c.gpu_copy_from(a.gpu_memory());

        detail::inplace_fft1_kernel(c, etl::size(c));
    } else {
        c = a;

        detail::inplace_fft1_kernel(c, etl::size(a));
    }
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param c The output expression
 */
template <typename C>
void inplace_fft1(C&& c) {
    static_assert(is_complex<C>);

    detail::inplace_fft1_kernel(c, etl::size(c));
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft1_many(A&& a, C&& c) {
    static constexpr size_t N = decay_traits<A>::dimensions();

    size_t n     = etl::dim<N - 1>(a); //Size of the transform
    size_t batch = etl::size(a) / n;   //Number of batch

    if constexpr (is_complex<A>) {
        a.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        c.gpu_copy_from(a.gpu_memory());

        detail::inplace_fft1_many_kernel(c, batch, n);
    } else {
        c = a;

        detail::inplace_fft1_many_kernel(c, batch, n);
    }
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_fft1_many(C&& c) {
    static_assert(is_complex<C>);

    static constexpr size_t N = decay_traits<C>::dimensions();

    size_t n     = etl::dim<N - 1>(c); //Size of the transform
    size_t batch = etl::size(c) / n;   //Number of batch

    detail::inplace_fft1_many_kernel(c, batch, n);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1(A&& a, C&& c) {
    a.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();
    c.gpu_copy_from(a.gpu_memory());

    detail::inplace_ifft1_kernel(c, etl::size(c));

    detail::scale_back(c);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param c The output expression
 */
template <typename C>
void inplace_ifft1(C&& c) {
    c.ensure_gpu_up_to_date();

    detail::inplace_ifft1_kernel(c, etl::size(c));

    detail::scale_back(c);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1_real(A&& a, C&& c) {
    auto tmp = force_temporary(a);

    detail::inplace_ifft1_kernel(tmp, etl::size(tmp));

    detail::scale_back_real(tmp, c);
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft1_many(A&& a, C&& c) {
    static constexpr size_t N = decay_traits<A>::dimensions();

    size_t n     = etl::dim<N - 1>(a); //Size of the transform
    size_t batch = etl::size(a) / n;   //Number of batch

    a.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();
    c.gpu_copy_from(a.gpu_memory());

    detail::inplace_ifft1_many_kernel(c, batch, n);

    detail::scale_back(c, 1.0 / double(n));
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_ifft1_many(C&& c) {
    static constexpr size_t N = decay_traits<C>::dimensions();

    size_t n     = etl::dim<N - 1>(c); //Size of the transform
    size_t batch = etl::size(c) / n;   //Number of batch

    c.ensure_gpu_up_to_date();

    detail::inplace_ifft1_many_kernel(c, batch, n);

    detail::scale_back(c, 1.0 / double(n));
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv1_full(A&& a, B&& b, C&& c) {
    using type = value_t<A>;

    decltype(auto) handle = start_cufft();

    const size_t size = etl::size(c);

    //Note: use of value_t to make the type dependent!
    dyn_vector<etl::complex<type>> a_padded(size);
    dyn_vector<etl::complex<type>> b_padded(size);

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    direct_copy(a.memory_start(), a.memory_end(), a_padded.memory_start());
    direct_copy(b.memory_start(), b.memory_end(), b_padded.memory_start());

    // FFT of the padded vectors

    detail::inplace_fft1_kernel(a_padded, size);
    detail::inplace_fft1_kernel(b_padded, size);

    // Element wise multiplication of the two vectors

    a_padded *= b_padded;

    // Inverse FFT of the result

    detail::inplace_ifft1_kernel(a_padded, size);

    // Scale back the real part

    a_padded.ensure_cpu_up_to_date();

    for (size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real * (1.0 / size);
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft2(A&& a, C&& c) {
    if constexpr (is_complex<A>) {
        a.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();
        c.gpu_copy_from(a.gpu_memory());

        detail::inplace_fft2_kernel(c, etl::dim<0>(c), etl::dim<1>(c));
    } else {
        c = a;

        detail::inplace_fft2_kernel(c, etl::dim<0>(a), etl::dim<1>(a));
    }
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_fft2(C&& c) {
    static_assert(is_complex<C>);

    c.ensure_gpu_up_to_date();

    detail::inplace_fft2_kernel(c, etl::dim<0>(c), etl::dim<1>(c));
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2(A&& a, C&& c) {
    a.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();
    c.gpu_copy_from(a.gpu_memory());

    detail::inplace_ifft2_kernel(c, etl::dim<0>(c), etl::dim<1>(c));

    detail::scale_back(c);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param c The output expression
 */
template <typename C>
void inplace_ifft2(C&& c) {
    c.ensure_gpu_up_to_date();

    detail::inplace_ifft2_kernel(c, etl::dim<0>(c), etl::dim<1>(c));

    detail::scale_back(c);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real(A&& a, C&& c) {
    auto tmp = force_temporary(a);

    detail::inplace_ifft2_kernel(tmp, etl::dim<0>(tmp), etl::dim<1>(tmp));

    detail::scale_back_real(tmp, c);
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft2_many(A&& a, C&& c) {
    static constexpr size_t N = decay_traits<A>::dimensions();

    size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    if constexpr (is_complex<A>) {
        a.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();
        c.gpu_copy_from(a.gpu_memory());

        detail::inplace_fft2_many_kernel(c, batch, n1, n2);
    } else {
        c = a;

        detail::inplace_fft2_many_kernel(c, batch, n1, n2);
    }
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_fft2_many(C&& c) {
    static constexpr size_t N = decay_traits<C>::dimensions();

    size_t n1    = etl::dim<N - 2>(c);       //Size of the transform
    size_t n2    = etl::dim<N - 1>(c);       //Size of the transform
    size_t batch = etl::size(c) / (n1 * n2); //Number of batch

    c.ensure_gpu_up_to_date();

    detail::inplace_fft2_many_kernel(c, batch, n1, n2);
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft2_many(A&& a, C&& c) {
    static constexpr size_t N = decay_traits<A>::dimensions();

    size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();
    c.gpu_copy_from(a.gpu_memory());

    detail::inplace_ifft2_many_kernel(c, batch, n1, n2);

    detail::scale_back(c, 1.0 / double(n1 * n2));
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_ifft2_many(C&& c) {
    static constexpr size_t N = decay_traits<C>::dimensions();

    size_t n1    = etl::dim<N - 2>(c);       //Size of the transform
    size_t n2    = etl::dim<N - 1>(c);       //Size of the transform
    size_t batch = etl::size(c) / (n1 * n2); //Number of batch

    c.ensure_gpu_up_to_date();

    detail::inplace_ifft2_many_kernel(c, batch, n1, n2);

    detail::scale_back(c, 1.0 / double(n1 * n2));
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full([[maybe_unused]] I&& a, [[maybe_unused]] K&& b, [[maybe_unused]] C&& c) {
    if constexpr (conv2_possible<I, K, C>) {
        using T = value_t<I>;

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(), T(0.0));
        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped([[maybe_unused]] I&& a, [[maybe_unused]] K&& b, [[maybe_unused]] C&& c) {
    if constexpr (conv2_possible<I, K, C>) {
        using T = value_t<I>;

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 2> prepared_b(etl::dim<0>(b), etl::dim<1>(b));

        std::copy(b.memory_start(), b.memory_end(), prepared_b.memory_start());

        prepared_b.fflip_inplace();

        detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), prepared_b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(),
                                  T(0.0));
        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename KK, typename C>
void conv2_full_multi([[maybe_unused]] I&& input, [[maybe_unused]] KK&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv2_possible<I, KK, C>) {
        using T = value_t<I>;

        const auto K = etl::dim<0>(kernel);

        const auto k1 = etl::dim<1>(kernel);
        const auto k2 = etl::dim<2>(kernel);

        const auto c1 = etl::dim<1>(conv);
        const auto c2 = etl::dim<2>(conv);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        for (size_t k = 0; k < K; ++k) {
            const auto k_s = k1 * k2;
            const auto c_s = c1 * c2;

            const T* b = kernel.memory_start() + k * k_s;
            T* c       = conv.memory_start() + k * c_s;

            detail::conv2_full_kernel(input.memory_start(), etl::dim<0>(input), etl::dim<1>(input), b, k1, k2, c, T(0.0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename KK, typename C>
void conv2_full_multi_flipped([[maybe_unused]] I&& input, [[maybe_unused]] KK&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv2_possible<I, KK, C>) {
        using T = value_t<I>;

        const auto K = etl::dim<0>(kernel);

        const auto k1 = etl::dim<1>(kernel);
        const auto k2 = etl::dim<2>(kernel);

        const auto c1 = etl::dim<1>(conv);
        const auto c2 = etl::dim<2>(conv);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();
        conv.ensure_cpu_up_to_date();

        for (size_t k = 0; k < K; ++k) {
            const auto k_s = k1 * k2;
            const auto c_s = c1 * c2;

            const T* b = kernel.memory_start() + k * k_s;
            T* c       = conv.memory_start() + k * c_s;

            etl::dyn_matrix<T, 2> prepared_b(k1, k2);

            std::copy(b, b + k_s, prepared_b.memory_start());

            prepared_b.fflip_inplace();

            detail::conv2_full_kernel(input.memory_start(), etl::dim<0>(input), etl::dim<1>(input), prepared_b.memory_start(), k1, k2, c, T(0.0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_full([[maybe_unused]] I&& input, [[maybe_unused]] KK&& kernel, [[maybe_unused]] CC&& conv) {
    if constexpr (conv2_possible<I, KK, CC>) {
        using T = value_t<I>;

        if (etl::dim<1>(kernel) > 0) {
            auto conv_i_inc = etl::dim<1>(conv) * etl::dim<2>(conv) * etl::dim<3>(conv);
            auto conv_c_inc = etl::dim<2>(conv) * etl::dim<3>(conv);

            auto kernel_k_inc = etl::dim<1>(kernel) * etl::dim<2>(kernel) * etl::dim<3>(kernel);
            auto kernel_c_inc = etl::dim<2>(kernel) * etl::dim<3>(kernel);

            auto input_i_inc = etl::dim<1>(input) * etl::dim<2>(input) * etl::dim<3>(input);
            auto input_k_inc = etl::dim<2>(input) * etl::dim<3>(input);

            const auto N = etl::dim<0>(input);
            const auto K = etl::dim<0>(kernel);
            const auto C = etl::dim<1>(kernel);

            const auto m1 = etl::dim<2>(input);
            const auto m2 = etl::dim<3>(input);

            const auto n1 = etl::dim<2>(kernel);
            const auto n2 = etl::dim<3>(kernel);

            const size_t s1   = m1 + n1 - 1;
            const size_t s2   = m2 + n2 - 1;
            const size_t size = s1 * s2;

            std::fill(conv.memory_start(), conv.memory_end(), 0);

            decltype(auto) handle = start_cufft();

            dyn_matrix<etl::complex<T>, 3> b_padded(K, C, size);
            std::fill(b_padded.memory_start(), b_padded.memory_end(), 0);

            dyn_matrix<etl::complex<T>, 3> a_padded(N, K, size);
            std::fill(a_padded.memory_start(), a_padded.memory_end(), 0);

            // Necessary for padding
            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            // Fully pad the inputs

            for (size_t i = 0; i < N; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    const T* a = input.memory_start() + i * input_i_inc + k * input_k_inc; // input(i)(k)

                    for (size_t ii = 0; ii < m1; ++ii) {
                        direct_copy_n(a + ii * m2, a_padded(i)(k).memory_start() + ii * s2, m2);
                    }
                }
            }

            // Fully pad the kernels

            for (size_t k = 0; k < etl::dim<0>(kernel); ++k) {
                for (size_t c = 0; c < etl::dim<1>(kernel); ++c) {
                    const T* b = kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc; // kernel(k)(c)

                    for (size_t i = 0; i < n1; ++i) {
                        direct_copy_n(b + i * n2, b_padded(k)(c).memory_start() + i * s2, n2);
                    }
                }
            }

            // They have been modified
            a_padded.invalidate_gpu();
            b_padded.invalidate_gpu();

            // Compute all the FFT of the inputs and kernels at once

            detail::inplace_fft2_many_kernel(a_padded, N * K, s1, s2);
            detail::inplace_fft2_many_kernel(b_padded, N * K, s1, s2);

            // Need CPU for multiplication
            a_padded.ensure_cpu_up_to_date();
            b_padded.ensure_cpu_up_to_date();

            // TODO For maximum performance
            // - tmp should have one more dimensions
            // - All the Inverse FFT should be done in one pass
            // - The multiplications and the conversion to real should be done in parallel

            dyn_matrix<etl::complex<T>, 3> tmp(C, K, size);

            for (size_t i = 0; i < N; ++i) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t k = 0; k < K; ++k) {
                        tmp(c)(k) = a_padded(i)(k) >> b_padded(k)(c);
                    }
                }

                // Perform the inverse FFT
                detail::inplace_ifft2_many_kernel(tmp, C * K, s1, s2);

                //  Need the CPU for scaling back
                tmp.ensure_cpu_up_to_date();

                for (size_t c = 0; c < C; ++c) {
                    for (size_t k = 0; k < K; ++k) {
                        T* cc = conv.memory_start() + i * conv_i_inc + c * conv_c_inc; // conv(i)(c)

                        for (size_t i = 0; i < size; ++i) {
                            cc[i] += tmp(c, k, i).real * (T(1.0) / size);
                        }
                    }
                }
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full_flipped([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv2_possible<I, K, C>) {
        using T = value_t<I>;

        if (etl::dim<1>(kernel) > 0) {
            kernel.ensure_cpu_up_to_date(); // Need for flipping

            etl::dyn_matrix<T, 4> prepared_k(etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel), etl::dim<3>(kernel));

            std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

            prepared_k.deep_fflip_inplace();
            prepared_k.invalidate_gpu();

            conv4_full(input, prepared_k, conv);
        }
    } else {
        cpp_unreachable("Invalid call to cufft::conv2_full");
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft1([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1_real([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft1_many([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft1_many([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft2([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft2_many([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft2_many([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_fft1([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_ifft1([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_ifft1_real([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_fft1_many([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_ifft1_many([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_fft2([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_ifft2([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename C>
void inplace_ifft2_real([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_fft2_many([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename C>
void inplace_ifft2_many([[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv1_full([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_flipped([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi_flipped([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full_flipped([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cufft fft");
}

    //COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::cufft
