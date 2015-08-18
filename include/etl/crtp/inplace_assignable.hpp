//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "etl/impl/transpose.hpp"
#include "etl/impl/fft.hpp"

/*
 * Use CRTP technique to inject inplace operations into expressions and value classes.
 */

namespace etl {

template<typename D>
struct inplace_assignable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    template<typename E>
    derived_t& scale_inplace(E&& e){
        as_derived() *= e;

        return as_derived();
    }

    derived_t& fflip_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() <= 2, "Impossible to fflip a matrix of D > 2");

        if(etl_traits<derived_t>::dimensions() == 2){
            std::reverse(as_derived().begin(), as_derived().end());
        }

        return as_derived();
    }

    template<typename S = D, cpp_enable_if((decay_traits<S>::dimensions() > 3))>
    derived_t& deep_transpose_inplace(){
        decltype(auto) mat = as_derived();

        for(std::size_t i = 0; i < etl::dim<0>(mat); ++i){
            mat(i).deep_transpose_inplace();
        }

        return mat;
    }

    template<typename S = D, cpp_enable_if((decay_traits<S>::dimensions() == 3))>
    derived_t& deep_transpose_inplace(){
        decltype(auto) mat = as_derived();

        for(std::size_t i = 0; i < etl::dim<0>(mat); ++i){
            mat(i).transpose_inplace();
        }

        return mat;
    }

    template<typename S = D, cpp_disable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");
        cpp_assert(etl::dim<0>(as_derived()) == etl::dim<1>(as_derived()), "Only square matrices can be tranposed inplace");

        detail::inplace_square_transpose<derived_t>::apply(as_derived());

        return as_derived();
    }

    template<typename S = D, cpp_enable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");

        decltype(auto) mat = as_derived();

        if(etl::dim<0>(mat) == etl::dim<1>(mat)){
            detail::inplace_square_transpose<derived_t>::apply(mat);
        } else {
            detail::inplace_rectangular_transpose<derived_t>::apply(mat);

            using std::swap;
            swap(mat.unsafe_dimension_access(0), mat.unsafe_dimension_access(1));
        }

        return mat;
    }

    derived_t& fft_inplace(){
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() == 1, "Only vector can use fft_inplace, use fft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::fft1_impl<derived_t, derived_t>::apply(mat, mat);

        return mat;
    }

    derived_t& ifft_inplace(){
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace IFFT");
        static_assert(etl_traits<derived_t>::dimensions() == 1, "Only vector can use ifft_inplace, use ifft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::ifft1_impl<derived_t, derived_t>::apply(mat, mat);

        return mat;
    }

    derived_t& fft2_inplace(){
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only vector can use fft_inplace, use fft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::fft2_impl<derived_t, derived_t>::apply(mat, mat);

        return mat;
    }

    derived_t& ifft2_inplace(){
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace IFFT");
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only vector can use ifft_inplace, use ifft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::ifft2_impl<derived_t, derived_t>::apply(mat, mat);

        return mat;
    }
};

} //end of namespace etl
