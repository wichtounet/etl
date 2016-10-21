//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Define traits to get vectorization information for types when no vector mode is available.
 */
template <typename T>
struct no_intrinsic_traits {
    static constexpr bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr std::size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr std::size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The intrinsic type
};

/*!
 * \brief Vectorization support when no vectorization is enabled.
 *
 * This class is purely here to ensure compilation, it will never be called at runtime
 */
struct no_vec {
    /*!
     * \brief The traits for this vectorization implementation
     */
    template <typename T>
    using traits = no_intrinsic_traits<T>;

    /*!
     * \brief The vector type for this vectorization implementation
     */
    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type;

    /*!
     * \brief Unaligned store value to memory
     * \param memory The target memory
     * \param value The value to store
     */
    template <typename F, typename M>
    static inline void storeu(F* memory, M value) {
        cpp_unused(memory);
        cpp_unused(value);
    }

    /*!
     * \brief Aligned store value to memory
     * \param memory The target memory
     * \param value The value to store
     */
    template <typename F, typename M>
    static inline void store(F* memory, M value) {
        cpp_unused(memory);
        cpp_unused(value);
    }

    /*!
     * \brief Aligned load a vector from memory
     * \param memory The target memory
     * \return Vector of values from memory
     */
    template <typename F>
    static F load(const F* memory) {
        cpp_unused(memory);
    }

    /*!
     * \brief Unaligned load a vector from memory
     * \param memory The target memory
     * \return Vector of values from memory
     */
    template <typename F>
    static F loadu(const F* memory) {
        cpp_unused(memory);
    }

    /*!
     * \brief Create a vector containing the given value
     * \param value The value
     * \return Vector of value
     */
    template <typename F>
    static F set(F value) {
        cpp_unused(value);
    }

    /*!
     * \brief Vector addition or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M add(M lhs, M rhs) {
        cpp_unused(lhs);
        cpp_unused(rhs);
    }

    /*!
     * \brief Vector subtraction or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M sub(M lhs, M rhs) {
        cpp_unused(lhs);
        cpp_unused(rhs);
    }

    /*!
     * \brief Vector multiplication or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M mul(M lhs, M rhs) {
        cpp_unused(lhs);
        cpp_unused(rhs);
    }

    /*!
     * \brief Vector division or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M div(M lhs, M rhs) {
        cpp_unused(lhs);
        cpp_unused(rhs);
    }

    /*!
     * \brief Vector square root
     * \param value The input values
     * \return The square root of the input values
     */
    template <typename M>
    static M sqrt(M value) {
        cpp_unused(value);
    }

    /*!
     * \brief Compute the negative value of the input
     * \param value The input values
     * \return The negative values of the input values
     */
    template <typename M>
    static M minus(M value) {
        cpp_unused(value);
    }
};

} //end of namespace etl
