//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace cufft {

#ifdef ETL_CUFFT_MODE

namespace detail {

template <typename T>
void inplace_cfft1_kernel(opaque_memory<T,1>& a_gpu, std::size_t n) {
    decltype(auto) handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T>
void inplace_zfft1_kernel(opaque_memory<T,1>& a_gpu, std::size_t n) {
    decltype(auto) handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_cfft1_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_C2C, batch);

    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_zfft1_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_cifft1_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_C2C, batch);

    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T, std::size_t D>
void inplace_zifft1_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t n) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T>
void inplace_cifft1_kernel(opaque_memory<T,1>& a_gpu, std::size_t n) {
    decltype(auto) handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T>
void inplace_zifft1_kernel(opaque_memory<T,1>& a_gpu, std::size_t n) {
    decltype(auto) handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T>
inline void inplace_cfft2_kernel(opaque_memory<T,2>& a_gpu, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T>
inline void inplace_zfft2_kernel(opaque_memory<T,2>& a_gpu, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_cfft2_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, CUFFT_C2C, batch);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_zfft2_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims,
                  nullptr, 1, d1 * d2,
                  nullptr, 1, d1 * d2,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);
}

template <typename T, std::size_t D>
void inplace_cifft2_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, CUFFT_C2C, batch);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T, std::size_t D>
void inplace_zifft2_many_kernel(opaque_memory<T,D>& a_gpu, std::size_t batch, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims,
                  nullptr, 1, d1 * d2,
                  nullptr, 1, d1 * d2,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T>
void inplace_cifft2_kernel(opaque_memory<T,2>& a_gpu, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

template <typename T>
void inplace_zifft2_kernel(opaque_memory<T,2>& a_gpu, std::size_t d1, std::size_t d2) {
    decltype(auto) handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_INVERSE);
}

inline cufftResult cufft_exec_c2c(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction){
    return cufftExecC2C(plan, idata, odata, direction);
}

inline cufftResult cufft_exec_c2c(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction){
    return cufftExecZ2Z(plan, idata, odata, direction);
}

template <typename T>
void conv2_full_kernel(const T* a, std::size_t m1, std::size_t m2, const T* b, std::size_t n1, std::size_t n2, T* c, T beta) {
    const std::size_t s1 = m1 + n1 - 1;
    const std::size_t s2 = m2 + n2 - 1;
    const std::size_t size = s1 * s2;

    decltype(auto) handle = start_cufft();

    dyn_vector<etl::complex<T>> a_padded(size);
    dyn_vector<etl::complex<T>> b_padded(size);

    for (std::size_t i = 0; i < m1; ++i) {
        direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
    }

    for (std::size_t i = 0; i < n1; ++i) {
        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
    }

    auto gpu_a = a_padded.direct();
    auto gpu_b = b_padded.direct();

    gpu_a.gpu_allocate_copy();
    gpu_b.gpu_allocate_copy();

    cufftPlan2d(&handle.get(), s1, s2, is_single_precision_t<T>::value ? CUFFT_C2C : CUFFT_Z2Z);

    cufft_exec_c2c(handle.get(), complex_cast(gpu_a.gpu_memory()), complex_cast(gpu_a.gpu_memory()), CUFFT_FORWARD);
    cufft_exec_c2c(handle.get(), complex_cast(gpu_b.gpu_memory()), complex_cast(gpu_b.gpu_memory()), CUFFT_FORWARD);

    gpu_a.gpu_copy_from();
    gpu_b.gpu_copy_from();

    a_padded *= b_padded;

    gpu_a.gpu_copy_to(); //Refresh the GPU memory

    cufft_exec_c2c(handle.get(), complex_cast(gpu_a.gpu_memory()), complex_cast(gpu_a.gpu_memory()), CUFFT_INVERSE);

    gpu_a.gpu_copy_from();

    if(beta == T(0.0)){
        for (std::size_t i = 0; i < size; ++i) {
            c[i] = a_padded[i].real * (T(1.0) / size);
        }
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            c[i] = beta * c[i] + a_padded[i].real * (T(1.0) / size);
        }
    }
}

} //End of namespace detail

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_kernel(c_gpu, etl::size(a));
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(c_gpu, etl::size(a));
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_kernel(a_gpu, etl::size(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(a_gpu, etl::size(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(c_gpu, batch, n);
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(c_gpu, batch, n);
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(a_gpu, batch, n);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(a_gpu, batch, n);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

template <typename C, cpp_enable_if(all_complex_single_precision<C>::value)>
void scale_back(C&& c, float factor) {
    auto c_gpu = c.direct();

#ifdef ETL_CUBLAS_MODE
    decltype(auto) handle = impl::cublas::start_cublas();

    cuComplex alpha = make_cuComplex(factor, 0.0);
    cublasCscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuComplex*>(c_gpu.gpu_memory()), 1);
#else
    //Copy from GPU and scale on CPU
    c_gpu.gpu_copy_from();
    c *= factor;

    //The GPU memory is not up-to-date => throw it away
    c_gpu.gpu_evict();
#endif
}

template <typename C, cpp_enable_if(all_complex_double_precision<C>::value)>
void scale_back(C&& c, double factor) {
    auto c_gpu = c.direct();

#ifdef ETL_CUBLAS_MODE
    decltype(auto) handle = impl::cublas::start_cublas();

    cuDoubleComplex alpha = make_cuDoubleComplex(factor, 0.0);
    cublasZscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuDoubleComplex*>(c_gpu.gpu_memory()), 1);
#else
    //Copy from GPU and scale on CPU
    c_gpu.gpu_copy_from();
    c *= factor;

    //The GPU memory is not up-to-date => throw it away
    c_gpu.gpu_evict();
#endif
}

template <typename C>
void scale_back(C&& c) {
    scale_back(std::forward<C>(c), 1.0 / etl::size(c));
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void scale_back_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

#ifdef ETL_CUBLAS_MODE
    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_if_necessary();

    decltype(auto) handle = impl::cublas::start_cublas();

    //Copy the real part of a to c
    cublasScopy(handle.get(), etl::size(c), reinterpret_cast<float*>(a_gpu.gpu_memory()), 2, reinterpret_cast<float*>(c_gpu.gpu_memory()), 1);

    //Scale c
    float alpha = 1.0 / etl::size(c);
    cublasSscal(handle.get(), etl::size(c), &alpha, c_gpu.gpu_memory(), 1);
#else
    auto tmp = allocate<std::complex<float>>(etl::size(a));

    cuda_check(cudaMemcpy(tmp.get(), a_gpu.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
#endif
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void scale_back_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

#ifdef ETL_CUBLAS_MODE
    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_if_necessary();

    decltype(auto) handle = impl::cublas::start_cublas();

    //Copy the real part of a to c
    cublasDcopy(handle.get(), etl::size(c), reinterpret_cast<double*>(a_gpu.gpu_memory()), 2, reinterpret_cast<double*>(c_gpu.gpu_memory()), 1);

    //Scale c
    double alpha = 1.0 / etl::size(c);
    cublasDscal(handle.get(), etl::size(c), &alpha, c_gpu.gpu_memory(), 1);
#else
    auto tmp = allocate<std::complex<double>>(etl::size(a));

    cuda_check(cudaMemcpy(tmp.get(), a_gpu.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
#endif
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a_gpu, etl::size(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a_gpu, etl::size(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a_gpu, etl::size(a));

    scale_back_real(a, c);
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a_gpu, etl::size(a));

    scale_back_real(a, c);
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_many_kernel(a_gpu, batch, n);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c, 1.0 / double(n));
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_many_kernel(a_gpu, batch, n);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c, 1.0 / double(n));
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

    const std::size_t size     = etl::size(c);

    //Note: use of value_t to make the type dependent!
    dyn_vector<etl::complex<type>> a_padded(size);
    dyn_vector<etl::complex<type>> b_padded(size);

    direct_copy(a.memory_start(), a.memory_end(), a_padded.memory_start());
    direct_copy(b.memory_start(), b.memory_end(), b_padded.memory_start());

    auto gpu_a = a_padded.direct();
    auto gpu_b = b_padded.direct();

    gpu_a.gpu_allocate_copy();
    gpu_b.gpu_allocate_copy();

    auto cufft_type = is_single_precision_t<type>::value ? CUFFT_C2C : CUFFT_Z2Z;
    cufftPlan1d(&handle.get(), size, cufft_type, 1);

    detail::cufft_exec_c2c(handle.get(), complex_cast(gpu_a.gpu_memory()), complex_cast(gpu_a.gpu_memory()), CUFFT_FORWARD);
    detail::cufft_exec_c2c(handle.get(), complex_cast(gpu_b.gpu_memory()), complex_cast(gpu_b.gpu_memory()), CUFFT_FORWARD);

    gpu_a.gpu_copy_from();
    gpu_b.gpu_copy_from();

    a_padded *= b_padded;

    gpu_a.gpu_copy_to(); //Refresh the GPU memory

    detail::cufft_exec_c2c(handle.get(), complex_cast(gpu_a.gpu_memory()), complex_cast(gpu_a.gpu_memory()), CUFFT_INVERSE);

    gpu_a.gpu_copy_from();

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real * (1.0 / size);
    }

    //Get rid of the GPU memory
    gpu_a.gpu_evict();
    gpu_b.gpu_evict();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_kernel(c_gpu, etl::dim<0>(a), etl::dim<1>(a));
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    auto c_gpu = c.direct();

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(c_gpu, etl::dim<0>(a), etl::dim<1>(a));
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    scale_back_real(a, c);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    auto a_gpu = a.direct();

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_kernel(a_gpu, etl::dim<0>(a), etl::dim<1>(a));

    scale_back_real(a, c);
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_many_kernel(c_gpu, batch, n1, n2);
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_many_kernel(c_gpu, batch, n1, n2);
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_many_kernel(a_gpu, batch, n1, n2);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_many_kernel(a_gpu, batch, n1, n2);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_many_kernel(a_gpu, batch, n1, n2);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c, 1.0 / double(n1 * n2));
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr std::size_t N = decay_traits<A>::dimensions();

    auto a_gpu = a.direct();
    auto c_gpu = c.direct();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a_gpu.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_many_kernel(a_gpu, batch, n1, n2);

    c_gpu.gpu_reallocate(a_gpu.gpu_release());

    scale_back(c, 1.0 / double(n1 * n2));
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv2_full(const opaque_memory<T, 2>& a, const opaque_memory<T, 2>& b, const opaque_memory<T, 2>& c) {
    detail::conv2_full_kernel(a.memory_start(), a.dim(0), a.dim(1), b.memory_start(), b.dim(0), b.dim(1), c.memory_start(), T(0.0));
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv2_full_flipped(const opaque_memory<T, 2>& a, const opaque_memory<T, 2>& b, const opaque_memory<T, 2>& c) {
    etl::dyn_matrix<T, 2> prepared_b(b.dim(0), b.dim(1));

    std::copy(b.memory_start(), b.memory_end(), prepared_b.memory_start());

    prepared_b.fflip_inplace();

    detail::conv2_full_kernel(a.memory_start(), a.dim(0), a.dim(1), prepared_b.memory_start(), b.dim(0), b.dim(1), c.memory_start(), T(0.0));
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv2_full_multi(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    const auto K = kernel.dim(0);

    const auto k1 = kernel.dim(1);
    const auto k2 = kernel.dim(2);

    const auto c1 = conv.dim(1);
    const auto c2 = conv.dim(2);

    for(size_t k = 0; k < K; ++k){
        const auto k_s = k1 * k2;
        const auto c_s = c1 * c2;

        const T* b = kernel.memory_start() + k * k_s;
        T* c       = conv.memory_start() + k * c_s;

        detail::conv2_full_kernel(input.memory_start(), input.dim(0), input.dim(1), b, k1, k2, c, T(0.0));
    }
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv2_full_multi_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    const auto K = kernel.dim(0);

    const auto k1 = kernel.dim(1);
    const auto k2 = kernel.dim(2);

    const auto c1 = conv.dim(1);
    const auto c2 = conv.dim(2);

    for(size_t k = 0; k < K; ++k){
        const auto k_s = k1 * k2;
        const auto c_s = c1 * c2;

        const T* b = kernel.memory_start() + k * k_s;
        T* c       = conv.memory_start() + k * c_s;

        etl::dyn_matrix<T, 2> prepared_b(k1, k2);

        std::copy(b, b + k_s, prepared_b.memory_start());

        prepared_b.fflip_inplace();

        detail::conv2_full_kernel(input.memory_start(), input.dim(0), input.dim(1), prepared_b.memory_start(), k1, k2, c, T(0.0));
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv4_full(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
    using detail::cufft_exec_c2c;

    if (kernel.dim(1) > 0) {
        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_k_inc = input.dim(2) * input.dim(3);

        const auto N = input.dim(0);
        const auto K = kernel.dim(0);
        const auto C = kernel.dim(1);

        const auto m1 = input.dim(2);
        const auto m2 = input.dim(3);

        const auto n1 = kernel.dim(2);
        const auto n2 = kernel.dim(3);

        const std::size_t s1   = m1 + n1 - 1;
        const std::size_t s2   = m2 + n2 - 1;
        const std::size_t size = s1 * s2;

        std::fill(conv.memory_start(), conv.memory_end(), 0);

        decltype(auto) handle = start_cufft();

        dyn_matrix<etl::complex<T>, 3> b_padded(K, C, size);
        std::fill(b_padded.memory_start(), b_padded.memory_end(), 0);

        dyn_matrix<etl::complex<T>, 3> a_padded(N, K, size);
        std::fill(a_padded.memory_start(), a_padded.memory_end(), 0);

        // Fully pad the inputs

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                const T* a = input.memory_start() + i * input_i_inc + k * input_k_inc;    // input(i)(k)

                for (std::size_t ii = 0; ii < m1; ++ii) {
                    direct_copy_n(a + ii * m2, a_padded(i)(k).memory_start() + ii * s2, m2);
                }
            }
        }

        // Fully pad the kernels

        for (std::size_t k = 0; k < kernel.dim(0); ++k) {
            for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                const T* b = kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc; // kernel(k)(c)

                for (std::size_t i = 0; i < n1; ++i) {
                    direct_copy_n(b + i * n2, b_padded(k)(c).memory_start() + i * s2, n2);
                }
            }
        }

        // Compute all the FFT of the inputs at once


        auto cufft_type = is_single_precision_t<T>::value ? CUFFT_C2C : CUFFT_Z2Z;

        {
            auto a_gpu = a_padded.direct();

            a_gpu.gpu_allocate_copy();

            int dims[] = {int(s1), int(s2)};

            cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, s1 * s2, nullptr, 1, s1 * s2, cufft_type, N * K);
            cufft_exec_c2c(handle.get(), complex_cast(a_gpu.gpu_memory()), complex_cast(a_gpu.gpu_memory()), CUFFT_FORWARD);

            a_gpu.gpu_copy_from();
        }

        // Compute all the FFT of the kernels at once

        {
            auto b_gpu = b_padded.direct();

            b_gpu.gpu_allocate_copy();

            int dims[] = {int(s1), int(s2)};

            cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, s1 * s2, nullptr, 1, s1 * s2, cufft_type, K * C);
            cufft_exec_c2c(handle.get(), complex_cast(b_gpu.gpu_memory()), complex_cast(b_gpu.gpu_memory()), CUFFT_FORWARD);

            b_gpu.gpu_copy_from();
        }

        // TODO For maximum performance
        // - tmp should have one more dimensions
        // - All the Inverse FFT should be done in one pass
        // - The multiplications and the conversion to real should be done in parallel

        dyn_matrix<etl::complex<T>, 3> tmp(C, K, size);

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t c = 0; c < C; ++c) {
                for (std::size_t k = 0; k < K; ++k) {
                    tmp(c)(k) = a_padded(i)(k) >> b_padded(k)(c);
                }
            }

            auto gpu_tmp = tmp.direct();
            gpu_tmp.gpu_allocate_copy();

            int dims[] = {int(s1), int(s2)};

            cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, s1 * s2, nullptr, 1, s1 * s2, cufft_type, C * K);
            cufft_exec_c2c(handle.get(), complex_cast(gpu_tmp.gpu_memory()), complex_cast(gpu_tmp.gpu_memory()), CUFFT_INVERSE);

            gpu_tmp.gpu_copy_from();

            for (std::size_t c = 0; c < C; ++c) {
                for (std::size_t k = 0; k < K; ++k) {
                    T* cc = conv.memory_start() + i * conv_i_inc + c * conv_c_inc; // conv(i)(c)

                    for (std::size_t i = 0; i < size; ++i) {
                        cc[i] += tmp(c, k, i).real * (T(1.0) / size);
                    }
                }
            }
        }
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename T>
void conv4_full_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
    if (kernel.dim(1) > 0) {
        etl::dyn_matrix<T, 4> prepared_k(kernel.dim(0), kernel.dim(1), kernel.dim(2), kernel.dim(3));

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        conv4_full(input, prepared_k.direct(), conv);
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
void fft1(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1_real(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
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
void fft1_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
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
void ifft1_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft2(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
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
void fft2_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
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
void ifft2_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv1_full(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
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
void conv2_full_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cufft fft");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cufft

} //end of namespace impl

} //end of namespace etl
