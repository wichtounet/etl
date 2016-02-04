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

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(c, etl::size(a));

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(c, batch, n);

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(c, batch, n);

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft1_many_kernel(a, batch, n);

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();

    c *= 1.0 / etl::size(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a, etl::size(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();

    c *= 1.0 / etl::size(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft1_kernel(a, etl::size(a));

    auto tmp = allocate<std::complex<float>>(etl::size(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft1_kernel(a, etl::size(a));

    auto tmp = allocate<std::complex<double>>(etl::size(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
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

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), c.memory_start());

    c.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(c, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cfft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zfft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_cifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();

    c *= 1.0 / etl::size(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    detail::inplace_zifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    c.gpu_reallocate(a.gpu_release());
    c.gpu_copy_from();

    c *= 1.0 / etl::size(c);
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    auto tmp = allocate<value_t<A>>(etl::size(a));

    detail::inplace_cifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    a.gpu_allocate_copy_if_necessary();

    auto tmp = allocate<value_t<A>>(etl::size(a));

    detail::inplace_zifft2_kernel(a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(tmp.get(), a.gpu_memory(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = tmp[i].real() / etl::size(a);
    }
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

    c.gpu_copy_from();
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

    c.gpu_copy_from();
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
    c.gpu_copy_from();
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
    c.gpu_copy_from();
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

template <typename A, typename C>
void fft1(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void ifft1(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void ifft1_real(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void fft1_many(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void fft2(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void ifft2(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void ifft2_real(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename C>
void fft2_many(A&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename B, typename C>
void fft1_convolve(A&& /*unused*/, B&& /*unused*/, C&& /*unused*/) {}

template <typename A, typename B, typename C>
void fft2_convolve(A&& /*unused*/, B&& /*unused*/, C&& /*unused*/) {}

#endif

} //end of namespace cufft

} //end of namespace impl

} //end of namespace etl
