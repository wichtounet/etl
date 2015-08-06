//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/config.hpp"
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

inline void inplace_cfft1_kernel(const impl::cuda::cuda_memory<std::complex<float>>& data, std::size_t n){
    auto handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_FORWARD);
}

inline void inplace_zfft1_kernel(const impl::cuda::cuda_memory<std::complex<double>>& data, std::size_t n){
    auto handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_FORWARD);
}

inline void inplace_cifft1_kernel(const impl::cuda::cuda_memory<std::complex<float>>& data, std::size_t n){
    auto handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_C2C, 1);
    cufftExecC2C(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_INVERSE);
}

inline void inplace_zifft1_kernel(const impl::cuda::cuda_memory<std::complex<double>>& data, std::size_t n){
    auto handle = start_cufft();

    cufftPlan1d(&handle.get(), n, CUFFT_Z2Z, 1);
    cufftExecZ2Z(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_INVERSE);
}

inline void inplace_cfft2_kernel(const impl::cuda::cuda_memory<std::complex<float>>& data, std::size_t d1, std::size_t d2){
    auto handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_FORWARD);
}

inline void inplace_zfft2_kernel(const impl::cuda::cuda_memory<std::complex<double>>& data, std::size_t d1, std::size_t d2){
    auto handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_FORWARD);
}

inline void inplace_cifft2_kernel(const impl::cuda::cuda_memory<std::complex<float>>& data, std::size_t d1, std::size_t d2){
    auto handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_C2C);
    cufftExecC2C(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_INVERSE);
}

inline void inplace_zifft2_kernel(const impl::cuda::cuda_memory<std::complex<double>>& data, std::size_t d1, std::size_t d2){
    auto handle = start_cufft();

    cufftPlan2d(&handle.get(), d1, d2, CUFFT_Z2Z);
    cufftExecZ2Z(handle.get(), complex_cast(data.get()), complex_cast(data.get()), CUFFT_INVERSE);
}

} //End of namespace detail

template<typename A, typename C>
void sfft1(A&& a, C&& c){
    std::copy(a.begin(), a.end(), c.begin());

    auto gpu_c = impl::cuda::cuda_allocate_copy(c);

    detail::inplace_cfft1_kernel(gpu_c, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void dfft1(A&& a, C&& c){
    std::copy(a.begin(), a.end(), c.begin());

    auto gpu_c = impl::cuda::cuda_allocate_copy(c);

    detail::inplace_zfft1_kernel(gpu_c, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void cfft1(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_cfft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void zfft1(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_zfft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void cifft1(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_cifft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);

    c *= 1.0 / etl::size(c);
}

template<typename A, typename C>
void zifft1(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_zifft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);

    c *= 1.0 / etl::size(c);
}

template<typename A, typename C>
void cifft1_real(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    auto tmp = allocate<std::complex<float>>(etl::size(a));

    detail::inplace_cifft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(tmp.get(), gpu_a.get(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template<typename A, typename C>
void zifft1_real(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    auto tmp = allocate<std::complex<double>>(etl::size(a));

    detail::inplace_zifft1_kernel(gpu_a, etl::size(a));

    cudaMemcpy(tmp.get(), gpu_a.get(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template<typename A, typename B, typename C>
void sfft1_convolve(A&& a, B&& b, C&& c){
    auto handle = start_cufft();

    const auto size = etl::size(c);
    const auto mem_size = size * sizeof(std::complex<float>);

    auto a_padded = allocate<std::complex<float>>(size);
    auto b_padded = allocate<std::complex<float>>(size);

    std::copy(a.begin(), a.end(), a_padded.get());
    std::copy(b.begin(), b.end(), b_padded.get());

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan1d(&handle.get(), size, CUFFT_C2C, 1);

    cufftExecC2C(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecC2C(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecC2C(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

template<typename A, typename B, typename C>
void dfft1_convolve(A&& a, B&& b, C&& c){
    auto handle = start_cufft();

    const auto size = etl::size(c);
    const auto mem_size = size * sizeof(std::complex<double>);

    auto a_padded = allocate<std::complex<double>>(size);
    auto b_padded = allocate<std::complex<double>>(size);

    std::copy(a.begin(), a.end(), a_padded.get());
    std::copy(b.begin(), b.end(), b_padded.get());

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan1d(&handle.get(), size, CUFFT_Z2Z, 1);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecZ2Z(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

template<typename A, typename C>
void sfft2(A&& a, C&& c){
    std::copy(a.begin(), a.end(), c.begin());

    auto gpu_c = impl::cuda::cuda_allocate_copy(c);

    detail::inplace_cfft2_kernel(gpu_c, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void dfft2(A&& a, C&& c){
    std::copy(a.begin(), a.end(), c.begin());

    auto gpu_c = impl::cuda::cuda_allocate_copy(c);

    detail::inplace_zfft2_kernel(gpu_c, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void cfft2(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_cfft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void zfft2(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_zfft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);
}

template<typename A, typename C>
void cifft2(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_cifft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);

    c *= 1.0 / etl::size(c);
}

template<typename A, typename C>
void zifft2(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    detail::inplace_zifft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(c.memory_start(), gpu_a.get(), etl::size(c) * sizeof(value_t<C>), cudaMemcpyDeviceToHost);

    c *= 1.0 / etl::size(c);
}

template<typename A, typename C>
void cifft2_real(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    auto tmp = allocate<value_t<A>>(etl::size(a));

    detail::inplace_cifft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(tmp.get(), gpu_a.get(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template<typename A, typename C>
void zifft2_real(A&& a, C&& c){
    auto gpu_a = impl::cuda::cuda_allocate_copy(a);

    auto tmp = allocate<value_t<A>>(etl::size(a));

    detail::inplace_zifft2_kernel(gpu_a, etl::dim<0>(a), etl::dim<1>(a));

    cudaMemcpy(tmp.get(), gpu_a.get(), etl::size(c) * sizeof(value_t<A>), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        c[i] = tmp[i].real() / etl::size(a);
    }
}

template<typename A, typename B, typename C>
void sfft2_convolve(A&& a, B&& b, C&& c){
    const auto m1 = etl::dim<0>(a);
    const auto n1= etl::dim<0>(b);
    const auto s1 = m1 + n1 - 1;

    const auto m2 = etl::dim<1>(a);
    const auto n2= etl::dim<1>(b);
    const auto s2 = m2 + n2 - 1;

    auto handle = start_cufft();

    const auto size = etl::size(c);
    const auto mem_size = size * sizeof(std::complex<float>);

    auto a_padded = allocate<std::complex<float>>(size);
    auto b_padded = allocate<std::complex<float>>(size);

    for(std::size_t i = 0; i < m1; ++i){
        for(std::size_t j = 0; j < m2; ++j){
            a_padded[i * s2 + j] = a(i,j);
        }
    }

    for(std::size_t i = 0; i < n1; ++i){
        for(std::size_t j = 0; j < n2; ++j){
            b_padded[i * s2 + j] = b(i,j);
        }
    }

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan2d(&handle.get(), s1, s2, CUFFT_C2C);

    cufftExecC2C(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecC2C(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecC2C(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        c[i] = a_padded[i].real() * (1.0f / size);
    }
}

template<typename A, typename B, typename C>
void dfft2_convolve(A&& a, B&& b, C&& c){
    const auto m1 = etl::dim<0>(a);
    const auto n1= etl::dim<0>(b);
    const auto s1 = m1 + n1 - 1;

    const auto m2 = etl::dim<1>(a);
    const auto n2= etl::dim<1>(b);
    const auto s2 = m2 + n2 - 1;

    auto handle = start_cufft();

    const auto size = etl::size(c);
    const auto mem_size = size * sizeof(std::complex<double>);

    auto a_padded = allocate<std::complex<double>>(size);
    auto b_padded = allocate<std::complex<double>>(size);

    for(std::size_t i = 0; i < m1; ++i){
        for(std::size_t j = 0; j < m2; ++j){
            a_padded[i * s2 + j] = a(i,j);
        }
    }

    for(std::size_t i = 0; i < n1; ++i){
        for(std::size_t j = 0; j < n2; ++j){
            b_padded[i * s2 + j] = b(i,j);
        }
    }

    auto gpu_a = impl::cuda::cuda_allocate_copy(a_padded.get(), size);
    auto gpu_b = impl::cuda::cuda_allocate_copy(b_padded.get(), size);

    cufftPlan2d(&handle.get(), s1, s2, CUFFT_Z2Z);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_a.get()), complex_cast(gpu_a.get()), CUFFT_FORWARD);
    cufftExecZ2Z(handle.get(), complex_cast(gpu_b.get()), complex_cast(gpu_b.get()), CUFFT_FORWARD);

    cudaMemcpy(a_padded.get(), gpu_a.get(), mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_padded.get(), gpu_b.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        a_padded[i] *= b_padded[i];
    }

    auto gpu_c = impl::cuda::cuda_allocate_copy(a_padded.get(), size);

    cufftExecZ2Z(handle.get(), complex_cast(gpu_c.get()), complex_cast(gpu_c.get()), CUFFT_INVERSE);

    cudaMemcpy(a_padded.get(), gpu_c.get(), mem_size, cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < size; ++i){
        c[i] = a_padded[i].real() * (1.0 / size);
    }
}

#else

template<typename A, typename C>
void sfft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void dfft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cfft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zfft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cifft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zifft1(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cifft1_real(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zifft1_real(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename B, typename C>
void sfft1_convolve(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename B, typename C>
void dfft1_convolve(A&& /*unused*/, B&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void sfft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void dfft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cfft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zfft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cifft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zifft2(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void cifft2_real(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename C>
void zifft2_real(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename B, typename C>
void sfft2_convolve(A&& /*unused*/, C&& /*unused*/);

template<typename A, typename B, typename C>
void dfft2_convolve(A&& /*unused*/, B&& /*unused*/, C&& /*unused*/);

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl
