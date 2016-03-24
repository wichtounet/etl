//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/allocator.hpp"

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

template<typename A>
void inplace_cfft1_kernel(A&& a, std::size_t n) {
    cufft_handle handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_zfft1_kernel(A&& a, std::size_t n) {
    cufft_handle handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_cfft1_many_kernel(A&& a, std::size_t batch, std::size_t n) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_C2C, batch);

    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_zfft1_many_kernel(A&& a, std::size_t batch, std::size_t n) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_cifft1_many_kernel(A&& a, std::size_t batch, std::size_t n) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_C2C, batch);

    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_zifft1_many_kernel(A&& a, std::size_t batch, std::size_t n) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(n)};

    cufftPlanMany(&handle.get(), 1, dims,
                  nullptr, 1, n,
                  nullptr, 1, n,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_cifft1_kernel(A&& a, std::size_t n) {
    cufft_handle handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_zifft1_kernel(A&& a, std::size_t n) {
    cufft_handle handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
inline void inplace_cfft2_kernel(A&& a, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
inline void inplace_zfft2_kernel(A&& a, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_cfft2_many_kernel(A&& a, std::size_t batch, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, CUFFT_C2C, batch);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_zfft2_many_kernel(A&& a, std::size_t batch, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims,
                  nullptr, 1, d1 * d2,
                  nullptr, 1, d1 * d2,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_FORWARD);
}

template<typename A>
void inplace_cifft2_many_kernel(A&& a, std::size_t batch, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims, nullptr, 1, d1 * d2, nullptr, 1, d1 * d2, CUFFT_C2C, batch);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_zifft2_many_kernel(A&& a, std::size_t batch, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    int dims[] = {int(d1), int(d2)};

    cufftPlanMany(&handle.get(), 2, dims,
                  nullptr, 1, d1 * d2,
                  nullptr, 1, d1 * d2,
                  CUFFT_Z2Z, batch);

    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_cifft2_kernel(A&& a, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

template<typename A>
void inplace_zifft2_kernel(A&& a, std::size_t d1, std::size_t d2) {
    cufft_handle handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(a.gpu_memory()), complex_cast(a.gpu_memory()), CUFFT_INVERSE);
}

} //End of namespace detail

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_kernel(c, etl::size(a));
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(c, etl::size(a));
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(c, batch, n);
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(c, batch, n);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());
}

template <typename C, cpp_enable_if(all_complex_single_precision<C>::value)>
void scale_back(C&& c, float factor){
#ifdef ETL_CUBLAS_MODE
    impl::cublas::cublas_handle handle = impl::cublas::start_cublas();

    cuComplex alpha = make_cuComplex(factor, 0.0);
    cublasCscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuComplex*>(c.gpu_memory()), 1);
#else
    //Copy from GPU and scale on CPU
    c.gpu_copy_from();
    c *= factor;

    //The GPU memory is not up-to-date => throw it away
    c.gpu_evict();
#endif
}

template <typename C, cpp_enable_if(all_complex_double_precision<C>::value)>
void scale_back(C&& c, double factor){
#ifdef ETL_CUBLAS_MODE
    impl::cublas::cublas_handle handle = impl::cublas::start_cublas();

    cuDoubleComplex alpha = make_cuDoubleComplex(factor, 0.0);
    cublasZscal(handle.get(), etl::size(c), &alpha, reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1);
#else
    //Copy from GPU and scale on CPU
    c.gpu_copy_from();
    c *= factor;

    //The GPU memory is not up-to-date => throw it away
    c.gpu_evict();
#endif
}

template <typename C>
void scale_back(C&& c){
    scale_back(std::forward<C>(c), 1.0 / etl::size(c));
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void scale_back_real(A&& a, C&& c){
#ifdef ETL_CUBLAS_MODE
    c.gpu_allocate_if_necessary();

    impl::cublas::cublas_handle handle = impl::cublas::start_cublas();

    //Copy the real part of a to c
    cublasScopy(handle.get(), etl::size(c), reinterpret_cast<float*>(a.gpu_memory()), 2, reinterpret_cast<float*>(c.gpu_memory()), 1);

    //Scale c
    float alpha = 1.0 / etl::size(c);
    cublasSscal(handle.get(), etl::size(c), &alpha, c.gpu_memory(), 1);
#else
    auto tmp = allocate<std::complex<float>>(etl::size(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
#endif
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void scale_back_real(A&& a, C&& c){
#ifdef ETL_CUBLAS_MODE
    c.gpu_allocate_if_necessary();

    impl::cublas::cublas_handle handle = impl::cublas::start_cublas();

    //Copy the real part of a to c
    cublasDcopy(handle.get(), etl::size(c), reinterpret_cast<double*>(a.gpu_memory()), 2, reinterpret_cast<double*>(c.gpu_memory()), 1);

    //Scale c
    double alpha = 1.0 / etl::size(c);
    cublasDscal(handle.get(), etl::size(c), &alpha, c.gpu_memory(), 1);
#else
    auto tmp = allocate<std::complex<double>>(etl::size(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
#endif
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());

    scale_back(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());

    scale_back(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a, etl::size(a));

    scale_back_real(a, c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a, etl::size(a));

    scale_back_real(a, c);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());

    scale_back(c, 1.0 / double(n));
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());

    scale_back(c, 1.0 / double(n));
}


template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_convolve(A&& a, B&& b, C&& c) {
    auto handle = start_cufft();

    const std::size_t size     = etl::size(c);
    const std::size_t mem_size = size * sizeof(std::complex<float>);

    auto a_padded = allocate<std::complex<float>>(size);
    auto b_padded = allocate<std::complex<float>>(size);

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_padded.get());
    standard_evaluator::direct_copy(b.memory_start(), b.memory_end(), b_padded.get());

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan1d(&handle.get(), size, CUFFT_C2C, 1);

    cufftExecC2C(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecC2C(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecC2C(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_convolve(A&& a, B&& b, C&& c) {
    auto handle = start_cufft();

    const std::size_t size     = etl::size(c);
    const std::size_t mem_size = size * sizeof(std::complex<double>);

    auto a_padded = allocate<std::complex<double>>(size);
    auto b_padded = allocate<std::complex<double>>(size);

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_padded.get());
    standard_evaluator::direct_copy(b.memory_start(), b.memory_end(), b_padded.get());

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan1d(&handle.get(), size, CUFFT_Z2Z, 1);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecZ2Z(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_kernel(c, etl::dim<0>(a), etl::dim<1>(a));
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(c, etl::dim<0>(a), etl::dim<1>(a));
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());

    scale_back(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());

    scale_back(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    scale_back_real(a, c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    scale_back_real(a, c);
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_many_kernel(c, batch, n1, n2);
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_many_kernel(c, batch, n1, n2);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_many_kernel(a, batch, n1, n2);

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_many_kernel(a, batch, n1, n2);

    c.gpu_reallocate(a.gpu_release());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_many_kernel(a, batch, n1, n2);

    c.gpu_reallocate(a.gpu_release());

    scale_back(c, 1.0 / double(n1 * n2));
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_many_kernel(a, batch, n1, n2);

    c.gpu_reallocate(a.gpu_release());

    scale_back(c, 1.0 / double(n1 * n2));
}

template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m1 = etl::dim<0>(a);
    const std::size_t n1 = etl::dim<0>(b);
    const std::size_t s1 = m1 + n1 - 1;

    const std::size_t m2 = etl::dim<1>(a);
    const std::size_t n2 = etl::dim<1>(b);
    const std::size_t s2 = m2 + n2 - 1;

    auto handle = start_cufft();

    const std::size_t size     = etl::size(c);
    const std::size_t mem_size = size * sizeof(std::complex<float>);

    auto a_padded = allocate<std::complex<float>>(size);
    auto b_padded = allocate<std::complex<float>>(size);

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t j = 0; j < m2; ++j) {
            a_padded[i * s2 + j] = a(i, j);
        }
    }

    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            b_padded[i * s2 + j] = b(i, j);
        }
    }

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan2d(&handle.get(), s1, s2, CUFFT_C2C);

    cufftExecC2C(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecC2C(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecC2C(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real() * (1.0f / size);
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m1 = etl::dim<0>(a);
    const std::size_t n1 = etl::dim<0>(b);
    const std::size_t s1 = m1 + n1 - 1;

    const std::size_t m2 = etl::dim<1>(a);
    const std::size_t n2 = etl::dim<1>(b);
    const std::size_t s2 = m2 + n2 - 1;

    auto handle = start_cufft();

    const std::size_t size     = etl::size(c);
    const std::size_t mem_size = size * sizeof(std::complex<double>);

    auto a_padded = allocate<std::complex<double>>(size);
    auto b_padded = allocate<std::complex<double>>(size);

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t j = 0; j < m2; ++j) {
            a_padded[i * s2 + j] = a(i, j);
        }
    }

    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            b_padded[i * s2 + j] = b(i, j);
        }
    }

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan2d(&handle.get(), s1, s2, CUFFT_Z2Z);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecZ2Z(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

#else

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft1(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
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
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void fft1_convolve(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void fft2_convolve(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

#endif

} //end of namespace cufft

} //end of namespace impl

} //end of namespace etl
