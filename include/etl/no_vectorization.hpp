//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
    static constexpr bool vectorizable = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr size_t size       = 1;          ///< Numbers of elements done at once
    static constexpr size_t alignment  = alignof(T); ///< Necessary number of bytes of alignment for this type

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
    static inline void storeu([[maybe_unused]] F* memory, [[maybe_unused]] M value) {}

    /*!
     * \brief Aligned store value to memory
     * \param memory The target memory
     * \param value The value to store
     */
    template <typename F, typename M>
    static inline void store([[maybe_unused]] F* memory, [[maybe_unused]] M value) {}

    /*!
     * \brief Aligned load a vector from memory
     * \param memory The target memory
     * \return Vector of values from memory
     */
    template <typename F>
    static F load([[maybe_unused]] const F* memory) {
        return F();
    }

    /*!
     * \brief Unaligned load a vector from memory
     * \param memory The target memory
     * \return Vector of values from memory
     */
    template <typename F>
    static F loadu([[maybe_unused]] const F* memory) {
        return F();
    }

    /*!
     * \brief Create a vector containing the given value
     * \param value The value
     * \return Vector of value
     */
    template <typename F>
    static F set([[maybe_unused]] F value) {
        return F();
    }

    /*!
     * \brief Create a vector containing the rounded up values
     * \param x The value
     * \return Vector of value
     */
    template <typename F>
    static F round_up([[maybe_unused]] F x) {
        return F();
    }

    /*!
     * \brief Vector addition or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M add([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector subtraction or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M sub([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector multiplication of a and b and add the result to c
     * \param a The left hand side of the multiplication
     * \param b The right hand side of the multiplication
     * \param c The right hand side of the addition
     * \return Vector of the results
     */
    template <typename M>
    static M fmadd([[maybe_unused]] M a, [[maybe_unused]] M b, [[maybe_unused]] M c) {
        return M();
    }

    /*!
     * \brief Vector multiplication or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M mul([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector division or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M div([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector maximum or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M max([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector minimum or lhs and rhs
     * \param lhs The left hand side of the operation
     * \param rhs The right hand side of the operation
     * \return Vector of the results
     */
    template <typename M>
    static M min([[maybe_unused]] M lhs, [[maybe_unused]] M rhs) {
        return M();
    }

    /*!
     * \brief Vector square root
     * \param value The input values
     * \return The square root of the input values
     */
    template <typename M>
    static M sqrt([[maybe_unused]] M value) {
        return M();
    }

    /*!
     * \brief Compute the negative value of the input
     * \param value The input values
     * \return The negative values of the input values
     */
    template <typename M>
    static M minus([[maybe_unused]] M value) {
        return M();
    }

    /*!
     * \brief Perform an horizontal sum of the given vector
     */
    template <typename M>
    static M hadd([[maybe_unused]] M value) {
        return M();
    }

    /*!
     * \brief Return a vector type filled with zeroes of the correct type
     */
    template <typename T>
    static T zero() {
        return T();
    }
};

} //end of namespace etl
