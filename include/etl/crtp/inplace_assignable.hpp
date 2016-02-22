//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/transpose.hpp"
#include "etl/impl/fft.hpp"

/*!
 * \file inplace_assignable.hpp
 * \brief Use CRTP technique to inject inplace operations into expressions and value classes.
 */

namespace etl {

/*!
 * \brief CRTP class to inject inplace operations to matrix and vector structures.
 *
 * This CRTP class injects inplace FFT, Transposition, flipping and scaling.
 */
template <typename D>
struct inplace_assignable {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Scale the matrix by the factor e, in place.
     *
     * \param e The scaling factor.
     */
    template <typename E>
    derived_t& scale_inplace(E&& e) {
        as_derived() *= e;

        return as_derived();
    }

    /*!
     * \brief Flip the matrix horizontally and vertically, in place.
     */
    derived_t& fflip_inplace() {
        static_assert(etl_traits<derived_t>::dimensions() <= 2, "Impossible to fflip a matrix of D > 2");

        if (etl_traits<derived_t>::dimensions() == 2) {
            std::reverse(as_derived().begin(), as_derived().end());
        }

        return as_derived();
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if(is_dyn_matrix<S>::value && (etl_traits<S>::dimensions() > 3))>
    derived_t& deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).direct_deep_transpose_inplace();
        }

        static constexpr const std::size_t d = etl_traits<S>::dimensions();

        using std::swap;
        swap(mat.unsafe_dimension_access(d - 1), mat.unsafe_dimension_access(d - 2));

        return mat;
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if(is_dyn_matrix<S>::value && (etl_traits<S>::dimensions() == 3))>
    derived_t& deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).direct_transpose_inplace();
        }

        static constexpr const std::size_t d = etl_traits<S>::dimensions();

        using std::swap;
        swap(mat.unsafe_dimension_access(d - 1), mat.unsafe_dimension_access(d - 2));

        return mat;
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if(!is_dyn_matrix<S>::value && (etl_traits<S>::dimensions() > 3))>
    derived_t& deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).deep_transpose_inplace();
        }

        return mat;
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if(!is_dyn_matrix<S>::value && (etl_traits<S>::dimensions() == 3))>
    derived_t& deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).transpose_inplace();
        }

        return mat;
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if((etl_traits<S>::dimensions() > 3))>
    derived_t& direct_deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).direct_deep_transpose_inplace();
        }

        return mat;
    }

    /*!
     * \brief Transpose each sub 2D matrix in place.
     */
    template <typename S = D, cpp_enable_if((etl_traits<S>::dimensions() == 3))>
    derived_t& direct_deep_transpose_inplace() {
        decltype(auto) mat = as_derived();

        for (std::size_t i = 0; i < etl::dim<0>(mat); ++i) {
            mat(i).direct_transpose_inplace();
        }

        return mat;
    }

    /*!
     * \brief Transpose the matrix in place.
     *
     * Only square fast matrix can be transpose in place, dyn matrix don't have any limitation.
     */
    template <typename S = D, cpp_disable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace() {
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");
        cpp_assert(etl::dim<0>(as_derived()) == etl::dim<1>(as_derived()), "Only square fast matrices can be tranposed inplace");

        detail::inplace_square_transpose<derived_t>::apply(as_derived());

        return as_derived();
    }

    /*!
     * \brief Transpose the matrix in place.
     *
     * Only square fast matrix can be transpose in place, dyn matrix don't have any limitation.
     */
    template <typename S = D, cpp_enable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace() {
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");

        decltype(auto) mat = as_derived();

        if (etl::dim<0>(mat) == etl::dim<1>(mat)) {
            detail::inplace_square_transpose<derived_t>::apply(mat);
        } else {
            detail::inplace_rectangular_transpose<derived_t>::apply(mat);

            using std::swap;
            swap(mat.unsafe_dimension_access(0), mat.unsafe_dimension_access(1));
        }

        return mat;
    }

    /*!
     * \brief Transpose the matrix in place.
     *
     * Only square fast matrix can be transpose in place, dyn matrix don't have any limitation.
     */
    derived_t& direct_transpose_inplace() {
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");

        decltype(auto) mat = as_derived();

        if (etl::dim<0>(mat) == etl::dim<1>(mat)) {
            detail::inplace_square_transpose<derived_t>::apply(mat);
        } else {
            detail::inplace_rectangular_transpose<derived_t>::apply(mat);
        }

        return mat;
    }

    /*!
     * \brief Perform inplace 1D FFT of the vector.
     */
    derived_t& fft_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() == 1, "Only vector can use fft_inplace, use fft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::fft1_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }

    /*!
     * \brief Perform many inplace 1D FFT of the matrix.
     *
     * This function considers the first dimension as being batches of 1D FFT.
     */
    derived_t& fft_many_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() > 1, "Only matrix of dimensions > 1 can use fft_many_inplace");

        decltype(auto) mat = as_derived();

        detail::fft1_many_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }

    /*!
     * \brief Perform inplace 1D Inverse FFT of the vector.
     */
    derived_t& ifft_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace IFFT");
        static_assert(etl_traits<derived_t>::dimensions() == 1, "Only vector can use ifft_inplace, use ifft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::ifft1_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }

    /*!
     * \brief Perform inplace 2D FFT of the matrix.
     */
    derived_t& fft2_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only matrix can use fft2_inplace, use fft_inplace for vectors");

        decltype(auto) mat = as_derived();

        detail::fft2_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }

    /*!
     * \brief Perform many inplace 2D FFT of the matrix.
     *
     * This function considers the first dimension as being batches of 2D FFT.
     */
    derived_t& fft2_many_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace FFT");
        static_assert(etl_traits<derived_t>::dimensions() > 2, "Only matrix of dimensions > 2 can use fft2_many_inplace");

        decltype(auto) mat = as_derived();

        detail::fft2_many_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }

    /*!
     * \brief Perform inplace 2D Inverse FFT of the matrix.
     */
    derived_t& ifft2_inplace() {
        static_assert(is_complex<derived_t>::value, "Only complex vector can use inplace IFFT");
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only vector can use ifft_inplace, use ifft2_inplace for matrices");

        decltype(auto) mat = as_derived();

        detail::ifft2_impl<derived_t, derived_t>::apply(mat, mat);

        mat.gpu_copy_from_if_necessary();

        return mat;
    }
};

} //end of namespace etl
