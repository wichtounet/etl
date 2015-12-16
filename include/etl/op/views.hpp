//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd> //For stream support

#include "cpp_utils/tmp.hpp"

#include "etl/tmp.hpp"

namespace etl {

template <typename T, typename S>
using return_helper =
    std::conditional_t<
        std::is_const<std::remove_reference_t<S>>::value,
        const value_t<T>&,
        std::conditional_t<
            cpp::and_u<
                std::is_lvalue_reference<S>::value,
                cpp::not_u<std::is_const<T>::value>::value>::value,
            value_t<T>&,
            value_t<T>>>;

template <typename T, typename S>
using const_return_helper = std::conditional_t<
    std::is_lvalue_reference<S>::value,
    const value_t<T>&,
    value_t<T>>;

template <typename T, std::size_t D>
struct dim_view {
    T sub;
    const std::size_t i;

    static_assert(D == 1 || D == 2, "Invalid dimension");

    using sub_type          = T;
    using value_type        = value_t<sub_type>;
    using memory_type       = memory_t<sub_type>;
    using const_memory_type = const_memory_t<sub_type>;
    using return_type       = return_helper<sub_type, decltype(sub(0, 0))>;
    using const_return_type = const_return_helper<sub_type, decltype(sub(0, 0))>;

    dim_view(sub_type sub, std::size_t i)
            : sub(sub), i(i) {}

    const_return_type operator[](std::size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    return_type operator[](std::size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    value_type read_flat(std::size_t j) const noexcept {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    const_return_type operator()(std::size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    return_type operator()(std::size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }
};

template <typename T>
struct sub_view {
    T sub;
    const std::size_t i;

    using sub_type       = T;
    using value_type        = value_t<sub_type>;
    using memory_type       = memory_t<sub_type>;
    using const_memory_type = const_memory_t<sub_type>;
    using return_type       = return_helper<sub_type, decltype(sub[0])>;
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>;

    sub_view(sub_type sub, std::size_t i)
            : sub(sub), i(i) {}

    const_return_type operator[](std::size_t j) const {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * subsize(sub) + j]
                   : sub[i + dim<0>(sub) * j];
    }

    return_type operator[](std::size_t j) {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * subsize(sub) + j]
                   : sub[i + dim<0>(sub) * j];
    }

    value_type read_flat(std::size_t j) const noexcept {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub.read_flat(i * subsize(sub) + j)
                   : sub.read_flat(i + dim<0>(sub) * j);
    }

    template <typename... S>
    const_return_type operator()(S... args) const {
        return sub(i, static_cast<std::size_t>(args)...);
    }

    template <typename... S>
    return_type operator()(S... args) {
        return sub(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }
};

namespace fast_matrix_view_detail {

template <typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

template <typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template <typename M, std::size_t I, typename S1>
inline constexpr std::size_t compute_index(S1 first) noexcept {
    return first;
}

template <typename M, std::size_t I, typename S1, typename... S, cpp_enable_if((sizeof...(S) > 0))>
inline constexpr std::size_t compute_index(S1 first, S... args) noexcept {
    return matrix_subsize<M, I>::value * first + compute_index<M, I + 1>(args...);
}

} //end of namespace fast_matrix_view_detail

template <typename T, std::size_t... Dims>
struct fast_matrix_view {
    T sub;

    using sub_type          = T;
    using value_type        = value_t<sub_type>;
    using memory_type       = memory_t<sub_type>;
    using const_memory_type = const_memory_t<sub_type>;
    using return_type       = return_helper<sub_type, decltype(sub[0])>;
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>;

    static constexpr std::size_t n_dimensions = sizeof...(Dims);

    explicit fast_matrix_view(sub_type sub)
            : sub(sub) {}

    template <typename... S>
    static constexpr std::size_t index(S... args) {
        return fast_matrix_view_detail::compute_index<fast_matrix_view<T, Dims...>, 0>(args...);
    }

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    return_type operator[](std::size_t j) {
        return sub[j];
    }

    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    template <typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), return_type&> operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    template <typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), const_return_type&> operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    template <std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return nth_size<D, 0, Dims...>::value;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

template <typename T>
struct dyn_vector_view {
    T sub;
    std::size_t rows;

    using sub_type          = T;
    using value_type        = value_t<sub_type>;
    using memory_type       = memory_t<sub_type>;
    using const_memory_type = const_memory_t<sub_type>;
    using return_type       = return_helper<sub_type, decltype(sub[0])>;
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>;

    dyn_vector_view(sub_type sub, std::size_t rows)
            : sub(sub), rows(rows) {}

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t j) const {
        return sub[j];
    }

    return_type operator[](std::size_t j) {
        return sub[j];
    }

    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    return_type operator()(std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

template <typename T>
struct dyn_matrix_view {
    T sub;
    std::size_t rows;
    std::size_t columns;

    using sub_type          = T;
    using value_type        = value_t<sub_type>;
    using memory_type       = memory_t<sub_type>;
    using const_memory_type = const_memory_t<sub_type>;
    using return_type       = return_helper<sub_type, decltype(sub[0])>;
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>;

    dyn_matrix_view(sub_type sub, std::size_t rows, std::size_t columns)
            : sub(sub), rows(rows), columns(columns) {}

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t i, std::size_t j) const {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * columns + j]
                   : sub[i + rows * j];
    }

    return_type operator[](std::size_t j) {
        return sub[j];
    }

    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    return_type operator()(std::size_t j) {
        return sub[j];
    }

    return_type operator()(std::size_t i, std::size_t j) {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * columns + j]
                   : sub[i + rows * j];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = true;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        if (D == 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_assert(d == 0, "Invalid dimension");
        cpp_unused(d);

        return size(v);
    }

    static constexpr std::size_t size() {
        return D == 1 ? etl_traits<sub_expr_t>::template dim<1>() : etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D2>
    static constexpr std::size_t dim() {
        static_assert(D2 == 0, "Invalid dimension");

        return size();
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for sub_view
 */
template <typename T>
struct etl_traits<etl::sub_view<T>> {
    using expr_t     = etl::sub_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = true;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;
    static constexpr const bool vectorizable            = has_direct_access<sub_expr_t>::value && storage_order == order::RowMajor;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub) / etl_traits<sub_expr_t>::dim(v.sub, 0);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d + 1);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

/*!
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, std::size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, Dims...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = true;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr std::size_t size(const expr_t& /*unused*/) {
        return mul_all<Dims...>::value;
    }

    static std::size_t dim(const expr_t& /*unused*/, std::size_t d) {
        return dyn_nth_size<Dims...>(d);
    }

    static constexpr std::size_t size() {
        return mul_all<Dims...>::value;
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return nth_size<D, 0, Dims...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(Dims);
    }
};

/*!
 * \brief Specialization for dyn_matrix_view.
 */
template <typename T>
struct etl_traits<etl::dyn_matrix_view<T>> {
    using expr_t     = etl::dyn_matrix_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = true;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.rows * v.columns;
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d == 0 ? v.rows : v.columns;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_vector_view.
 */
template <typename T>
struct etl_traits<etl::dyn_vector_view<T>> {
    using expr_t     = etl::dyn_vector_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = true;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.rows;
    }

    static std::size_t dim(const expr_t& v, std::size_t /*d*/) {
        return v.rows;
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};


template <typename T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dim_view<T, D>& v) {
    return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const sub_view<T>& v) {
    return os << "sub(" << v.sub << ", " << v.i << ")";
}

template <typename T, std::size_t Rows, std::size_t Columns>
std::ostream& operator<<(std::ostream& os, const fast_matrix_view<T, Rows, Columns>& v) {
    return os << "reshape[" << Rows << "," << Columns << "](" << v.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_view<T>& v) {
    return os << "reshape[" << v.rows << "," << v.columns << "](" << v.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const dyn_vector_view<T>& v) {
    return os << "reshape[" << v.rows << "](" << v.sub << ")";
}

} //end of namespace etl
