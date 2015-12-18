//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <random>
#include <functional>
#include <ctime>

#include "etl/math.hpp"

namespace etl {

/*!
 * \brief The random engine used by the noise operators
 */
using random_engine = std::mt19937_64;

template <typename T>
struct plus_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = true;  ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true;  ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param lhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs + rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::add(ymm1, ymm2);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "+";
    }
};

template <typename T>
struct minus_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = true; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param lhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs - rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::sub(ymm1, ymm2);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "-";
    }
};

template <typename T>
struct mul_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = vector_mode == vector_mode_t::AVX512 ? !is_complex_t<T>::value : true ; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param lhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs * rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::template mul<is_complex_t<T>::value>(ymm1, ymm2);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "*";
    }
};

template <typename T>
struct div_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = vector_mode == vector_mode_t::AVX512 ? !is_complex_t<T>::value : true ; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param lhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs / rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        const vec_type<V> ymm1(lhs);
        const vec_type<V> ymm2(rhs);
        return V::template div<is_complex_t<T>::value>(ymm1, ymm2);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "/";
    }
};

template <typename T>
struct mod_binary_op {
    static constexpr const bool vectorizable = false; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param lhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs % rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "%";
    }
};

template <typename T, typename E>
struct ranged_noise_binary_op {
    static constexpr const bool vectorizable = false; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static T apply(const T& x, E value) {
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);
        static auto noise = std::bind(normal_distribution, rand_engine);

        if (x == 0.0 || x == value) {
            return x;
        } else {
            return x + noise();
        }
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

template <typename T, typename E>
struct max_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::max(x, value);
    }

#ifdef __INTEL_COMPILER
    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::max(lhs, rhs);
    }
#endif

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "max";
    }
};

template <typename T, typename E>
struct min_binary_op {
    template<typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr const bool vectorizable = intel_compiler && !is_complex_t<T>::value; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::min(x, value);
    }

#ifdef __INTEL_COMPILER
    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template<typename V = default_vec>
    static cpp14_constexpr vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::min(lhs, rhs);
    }
#endif

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "min";
    }
};

template <typename T, typename E>
struct pow_binary_op {
    static constexpr const bool vectorizable = false; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::pow(x, value);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "pow";
    }
};

template <typename T, typename E>
struct one_if_binary_op {
    static constexpr const bool vectorizable = false; ///< Indicates if the opeator is vectorizable or not
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear or not
    static constexpr const bool desc_func    = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return 1.0 ? x == value : 0.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "one_if";
    }
};

} //end of namespace etl
