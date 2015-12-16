//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd> //For stream support

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/tmp.hpp"

namespace etl {

template <typename T, std::size_t... D>
struct rep_r_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static constexpr const std::size_t sub_d      = decay_traits<sub_type>::dimensions();
    static constexpr const std::size_t dimensions = sizeof...(D) + sub_d;

    sub_type sub;

    explicit rep_r_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sub[i / mul_all<D...>::value];
    }

    value_type read_flat(std::size_t i) const noexcept {
        return sub.read_flat(i / mul_all<D...>::value);
    }

    template <typename... Sizes, cpp_enable_if((sizeof...(Sizes) == dimensions))>
    value_type operator()(Sizes... sizes) const {
        return selected_only(std::make_index_sequence<sub_d>(), sizes...);
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

private:
    template <typename... Sizes, std::size_t... I>
    value_type selected_only(const std::index_sequence<I...>& /*seq*/, Sizes... sizes) const {
        return sub(cpp::nth_value<I>(sizes...)...);
    }
};

template <typename T, std::size_t... D>
struct rep_l_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static constexpr const std::size_t sub_d      = decay_traits<sub_type>::dimensions();
    static constexpr const std::size_t dimensions = sizeof...(D) + sub_d;

    sub_type sub;

    explicit rep_l_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sub[i % size(sub)];
    }

    value_type read_flat(std::size_t i) const noexcept {
        return sub.read_flat(i % size(sub));
    }

    template <typename... Sizes, cpp_enable_if((sizeof...(Sizes) == dimensions))>
    value_type operator()(Sizes... sizes) const {
        return selected_only(make_index_range<sizeof...(D), dimensions>(), sizes...);
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

private:
    template <typename... Sizes, std::size_t... I>
    value_type selected_only(const std::index_sequence<I...>& /*seq*/, Sizes... sizes) const {
        return sub(cpp::nth_value<I>(sizes...)...);
    }
};

template <typename T, std::size_t D>
struct dyn_rep_r_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static constexpr const std::size_t sub_d      = decay_traits<sub_type>::dimensions();
    static constexpr const std::size_t dimensions = D + sub_d;

    sub_type sub;
    std::array<std::size_t, D> reps;
    std::size_t m;

    dyn_rep_r_transformer(sub_type vec, std::array<std::size_t, D> reps_a)
            : sub(vec), reps(reps_a) {
        m = std::accumulate(reps.begin(), reps.end(), 1UL, [](std::size_t a, std::size_t b) { return a * b; });
    }

    value_type operator[](std::size_t i) const {
        return sub(i / m);
    }

    value_type read_flat(std::size_t i) const noexcept {
        return sub(i / m);
    }

    template <typename... Sizes, cpp_enable_if((sizeof...(Sizes) == dimensions))>
    value_type operator()(Sizes... sizes) const {
        return selected_only(std::make_index_sequence<sub_d>(), sizes...);
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

private:
    template <typename... Sizes, std::size_t... I>
    value_type selected_only(const std::index_sequence<I...>& /*seq*/, Sizes... sizes) const {
        return sub(cpp::nth_value<I>(sizes...)...);
    }
};

template <typename T, std::size_t D>
struct dyn_rep_l_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static constexpr const std::size_t sub_d      = decay_traits<sub_type>::dimensions();
    static constexpr const std::size_t dimensions = D + sub_d;

    sub_type sub;
    std::array<std::size_t, D> reps;
    std::size_t m;

    dyn_rep_l_transformer(sub_type vec, std::array<std::size_t, D> reps_a)
            : sub(vec), reps(reps_a) {
        m = std::accumulate(reps.begin(), reps.end(), 1UL, [](std::size_t a, std::size_t b) { return a * b; });
    }

    value_type operator[](std::size_t i) const {
        return sub(i % size(sub));
    }

    value_type read_flat(std::size_t i) const {
        return sub(i % size(sub));
    }

    template <typename... Sizes, cpp_enable_if((sizeof...(Sizes) == dimensions))>
    value_type operator()(Sizes... sizes) const {
        return selected_only(make_index_range<D, dimensions>(), sizes...);
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

private:
    template <typename... Sizes, std::size_t... I>
    value_type selected_only(const std::index_sequence<I...>& /*seq*/, Sizes... sizes) const {
        return sub(cpp::nth_value<I>(sizes...)...);
    }
};

template <typename T>
struct sum_r_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit sum_r_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sum(sub(i));
    }

    value_type read_flat(std::size_t i) const {
        return sum(sub(i));
    }

    template <typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return sum(sub(i));
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
};

template <typename T>
struct mean_r_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit mean_r_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return mean(sub(i));
    }

    value_type read_flat(std::size_t i) const {
        return mean(sub(i));
    }

    template <typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return mean(sub(i));
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
};

template <typename T>
struct sum_l_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit sum_l_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t j) const {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub[j + i * (size(sub) / dim<0>(sub))];
        }

        return m;
    }

    value_type read_flat(std::size_t j) const noexcept {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub.read_flat(j + i * (size(sub) / dim<0>(sub)));
        }

        return m;
    }

    template <typename... Sizes>
    value_type operator()(std::size_t j, Sizes... sizes) const {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub(i, j, sizes...);
        }

        return m;
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
};

template <typename T>
struct mean_l_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit mean_l_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t j) const {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub[j + i * (size(sub) / dim<0>(sub))];
        }

        return m / dim<0>(sub);
    }

    value_type read_flat(std::size_t j) const noexcept {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub.read_flat(j + i * (size(sub) / dim<0>(sub)));
        }

        return m / dim<0>(sub);
    }

    template <typename... Sizes>
    value_type operator()(std::size_t j, Sizes... sizes) const {
        value_type m = 0.0;

        for (std::size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub(i, j, sizes...);
        }

        return m / dim<0>(sub);
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
};

template <typename T>
struct hflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit hflip_transformer(sub_type vec)
            : sub(vec) {}

    static constexpr const bool matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    template <bool C = matrix, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        return sub[size(sub) - i - 1];
    }

    template <bool C = matrix, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub[i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j)];
    }

    template <bool C = matrix, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const {
        return sub.read_flat(size(sub) - i - 1);
    }

    template <bool C = matrix, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub.read_flat(i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j));
    }

    value_type operator()(std::size_t i) const {
        return sub(size(sub) - 1 - i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(i, columns(sub) - 1 - j);
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
};

template <typename T>
struct vflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit vflip_transformer(sub_type vec)
            : sub(vec) {}

    static constexpr const bool matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    template <bool C = matrix, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        return sub[i];
    }

    template <bool C = matrix, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub[(dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j];
    }

    template <bool C = matrix, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const {
        return sub.read_flat(i);
    }

    template <bool C = matrix, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub.read_flat((dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j);
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, j);
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
};

template <typename T>
struct fflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit fflip_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub[i];
        } else {
            return sub[size(sub) - i - 1];
        }
    }

    value_type read_flat(std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub.read_flat(i);
        } else {
            return sub.read_flat(size(sub) - i - 1);
        }
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, columns(sub) - 1 - j);
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
};

template <typename T>
struct transpose_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit transpose_transformer(sub_type vec)
            : sub(vec) {}

    value_type operator[](std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub[i];
        } else {
            if (decay_traits<sub_type>::storage_order == order::RowMajor) {
                std::size_t ii, jj;
                std::tie(ii, jj) = index_to_2d(sub, i);
                return sub(jj, ii);
            } else {
                return sub(i / dim<1>(sub), i % dim<1>(sub));
            }
        }
    }

    value_type read_flat(std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub.read_flat(i);
        } else {
            if (decay_traits<sub_type>::storage_order == order::RowMajor) {
                std::size_t ii, jj;
                std::tie(ii, jj) = index_to_2d(sub, i);
                return sub(jj, ii);
            } else {
                return sub(i / dim<1>(sub), i % dim<1>(sub));
            }
        }
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(j, i);
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
};

template <typename L, typename R>
struct mm_mul_transformer {
    using left_type  = L;
    using right_type = R;
    using value_type = value_t<L>;

    left_type left;
    right_type right;

    template <typename A, typename B, cpp_disable_if(all_fast<A, B>::value)>
    void check_mmul_sizes(const A& a, const B& b) {
        cpp_assert(
            dim<1>(a) == dim<0>(b) //interior dimensions
            ,
            "Invalid sizes for multiplication");
        cpp_unused(a);
        cpp_unused(b);
    }

    template <typename A, typename B, cpp_enable_if(all_fast<A, B>::value)>
    void check_mmul_sizes(const A& /*a*/, const B& /*b*/) {
        static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>() //interior dimensions
            ,
            "Invalid sizes for multiplication");
    }

    mm_mul_transformer(left_type left, right_type right)
            : left(left), right(right) {
        check_mmul_sizes(left, right);
    }

    value_type operator[](std::size_t i) const {
        std::size_t i_i, i_j;
        std::tie(i_i, i_j) = index_to_2d(left, i);
        return operator()(i_i, i_j);
    }

    value_type read_flat(std::size_t i) const {
        std::size_t i_i, i_j;
        std::tie(i_i, i_j) = index_to_2d(left, i);
        return operator()(i_i, i_j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        value_type c = 0;

        for (std::size_t k = 0; k < columns(left); k++) {
            c += left(i, k) * right(k, j);
        }

        return c;
    }

    left_type& lhs() {
        return left;
    }

    right_type& rhs() {
        return right;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return left.alias(rhs) || right.alias(rhs);
    }
};

template <typename T>
struct dyn_convmtx_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static_assert(decay_traits<T>::dimensions() == 1, "convmtx can only be applied on vectors");

    sub_type sub;
    std::size_t h;

    dyn_convmtx_transformer(sub_type sub, std::size_t h)
            : sub(sub), h(h) {}

    value_type operator[](std::size_t i) const {
        if (decay_traits<sub_type>::storage_order == order::RowMajor) {
            std::size_t i_i = i / (etl::size(sub) + h - 1);
            std::size_t i_j = i % (etl::size(sub) + h - 1);
            return operator()(i_i, i_j);
        } else {
            std::size_t i_i = i % h;
            std::size_t i_j = i / h;
            return operator()(i_i, i_j);
        }
    }

    value_type read_flat(std::size_t i) const {
        return operator[](i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        if (j < i) {
            return value_type(0);
        } else if (j >= etl::size(sub) + i) {
            return value_type(0);
        } else {
            return sub(j - i);
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
};

template <typename T>
struct dyn_convmtx2_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    static_assert(decay_traits<T>::dimensions() == 2, "convmtx2 can only be applied on matrices");

    sub_type sub;
    const std::size_t k1;
    const std::size_t k2;
    std::size_t i1;
    std::size_t i2;
    std::size_t inner_paddings;
    std::size_t inner_padding;

    dyn_convmtx2_transformer(sub_type sub, std::size_t k1, std::size_t k2)
            : sub(sub), k1(k1), k2(k2) {
        i1 = etl::dim<0>(sub);
        i2 = etl::dim<1>(sub);

        std::size_t c_height = (i1 + k1 - 1) * (i2 + k2 - 1);
        std::size_t c_width  = k1 * k2;

        auto max_fill  = c_height - ((i1 + k1 - 1) * ((c_width - 1) / k1) + (c_width - 1) % k1);
        inner_paddings = max_fill - (i1 * i2);
        inner_padding  = inner_paddings / (i2 - 1);
    }

    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / (k1 * k2);
        std::size_t i_j = i % (k1 * k2);
        return (*this)(i_i, i_j);
    }

    value_type read_flat(std::size_t i) const noexcept {
        std::size_t i_i = i / (k1 * k2);
        std::size_t i_j = i % (k1 * k2);
        return (*this)(i_i, i_j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        auto top_padding = (i1 + k1 - 1) * (j / k1) + j % k1;

        if (i < top_padding || i >= top_padding + (i1 * i2) + inner_paddings) {
            return value_type(0);
        } else {
            auto inner = i - top_padding;
            auto col   = inner % (i1 + inner_padding);
            auto block = inner / (i1 + inner_padding);

            if (col >= i1) {
                return value_type(0);
            } else {
                return sub(col, block);
            }
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
};

//TODO Make a temporary_expr out of this

template <typename A, typename M>
void convmtx2_direct_t(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const std::size_t c_height = (i1 + k1 - 1) * (i2 + k2 - 1);
    const std::size_t c_width  = k1 * k2;

    const auto max_fill       = c_height - ((i1 + k1 - 1) * ((c_width - 1) / k1) + (c_width - 1) % k1);
    const auto inner_paddings = max_fill - (i1 * i2);
    const auto inner_padding  = inner_paddings / (i2 - 1);

    auto* __restrict mm = m.memory_start();
    auto* __restrict ss = sub.memory_start();

    std::fill(mm, mm + etl::size(m), 0.0);

    for (std::size_t j = 0; j < c_width; ++j) {
        std::size_t big_i = (i1 + k1 - 1) * (j / k1) + j % k1;

        for (std::size_t ii = 0; ii < etl::dim<1>(sub); ++ii) {
            for (std::size_t jj = 0; jj < etl::dim<0>(sub); ++jj) {
                mm[j * c_width + big_i] = ss[jj * i2 + ii];
                ++big_i;
            }
            big_i += inner_padding;
        }
    }
}

//TODO Adapt this to an expression

template <typename A, typename M, cpp_disable_if(has_direct_access<A>::value&& has_direct_access<M>::value)>
void im2col_direct(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const std::size_t m_width = (i1 - k1 + 1) * (i2 - k2 + 1);

    for (std::size_t b = 0; b < m_width; ++b) {
        auto s_i = b % (i1 - k1 + 1);
        auto s_j = b / (i1 - k1 + 1);

        for (std::size_t b_i = 0; b_i < k1; ++b_i) {
            for (std::size_t b_j = 0; b_j < k2; ++b_j) {
                m(b_j * k1 + b_i, b) = sub(s_i + b_i, s_j + b_j);
            }
        }
    }
}

//TODO Find out why clang is unable to optimize the first version and then remove the second version

template <typename A, typename M, cpp_enable_if(has_direct_access<A>::value&& has_direct_access<M>::value)>
void im2col_direct(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const auto m_width = (i1 - k1 + 1) * (i2 - k2 + 1);

    const auto mm = m.memory_start();
    const auto ss = sub.memory_start();

    for (std::size_t b = 0; b < m_width; ++b) {
        auto s_i = b % (i1 - k1 + 1);
        auto s_j = b / (i1 - k1 + 1);

        for (std::size_t b_i = 0; b_i < k1; ++b_i) {
            for (std::size_t b_j = 0; b_j < k2; ++b_j) {
                mm[(b_j * k1 + b_i) * m_width + b] = ss[(s_i + b_i) * i2 + s_j + b_j];
            }
        }
    }
}

template <typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub;

    explicit p_max_pool_transformer(sub_type vec)
            : sub(vec) {}

    value_type pool(std::size_t i, std::size_t j) const {
        value_type p = 0;

        auto start_ii = (i / C1) * C1;
        auto start_jj = (j / C2) * C2;

        for (std::size_t ii = start_ii; ii < start_ii + C1; ++ii) {
            for (std::size_t jj = start_jj; jj < start_jj + C2; ++jj) {
                p += std::exp(sub(ii, jj));
            }
        }

        return p;
    }

    value_type pool(std::size_t k, std::size_t i, std::size_t j) const {
        value_type p = 0;

        auto start_ii = (i / C1) * C1;
        auto start_jj = (j / C2) * C2;

        for (std::size_t ii = start_ii; ii < start_ii + C1; ++ii) {
            for (std::size_t jj = start_jj; jj < start_jj + C2; ++jj) {
                p += std::exp(sub(k, ii, jj));
            }
        }

        return p;
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
};

template <typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_h_transformer : p_max_pool_transformer<T, C1, C2> {
    using base_type  = p_max_pool_transformer<T, C1, C2>; ///< The base type
    using sub_type   = typename base_type::sub_type;      ///< The type on which the expression works
    using value_type = typename base_type::value_type;    ///< The type of valuie

    using base_type::sub;

    static constexpr const bool d2d = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    explicit p_max_pool_h_transformer(sub_type vec)
            : base_type(vec) {}

    template <bool C = d2d, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return (*this)(i_i, i_j);
    }

    template <bool C = d2d, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i  = i / (dim<1>(sub) * dim<2>(sub));
        std::size_t i_ij = i % (dim<1>(sub) * dim<2>(sub));
        std::size_t i_j  = i_ij / dim<2>(sub);
        std::size_t i_k  = i_ij % dim<2>(sub);

        return (*this)(i_i, i_j, i_k);
    }

    template <bool C = d2d, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return (*this)(i_i, i_j);
    }

    template <bool C = d2d, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i  = i / (dim<1>(sub) * dim<2>(sub));
        std::size_t i_ij = i % (dim<1>(sub) * dim<2>(sub));
        std::size_t i_j  = i_ij / dim<2>(sub);
        std::size_t i_k  = i_ij % dim<2>(sub);

        return (*this)(i_i, i_j, i_k);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return std::exp(sub(i, j)) / (1.0 + base_type::pool(i, j));
    }

    value_type operator()(std::size_t k, std::size_t i, std::size_t j) const {
        return std::exp(sub(k, i, j)) / (1.0 + base_type::pool(k, i, j));
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
};

template <typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_p_transformer : p_max_pool_transformer<T, C1, C2> {
    using base_type  = p_max_pool_transformer<T, C1, C2>; ///< The base type
    using sub_type   = typename base_type::sub_type;      ///< The type on which the expression works
    using value_type = typename base_type::value_type;    ///< The type of valuie

    using base_type::sub;

    static constexpr const bool d2d = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    explicit p_max_pool_p_transformer(sub_type vec)
            : base_type(vec) {}

    template <bool C = d2d, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / (dim<1>(sub) / C2);
        std::size_t i_j = i % (dim<1>(sub) / C2);
        return (*this)(i_i, i_j);
    }

    template <bool C = d2d, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i  = i / ((dim<1>(sub) / C1) * (dim<2>(sub) / C2));
        std::size_t i_ij = i % ((dim<1>(sub) / C1) * (dim<2>(sub) / C2));
        std::size_t i_j  = i_ij / (dim<2>(sub) / C2);
        std::size_t i_k  = i_ij % (dim<2>(sub) / C2);

        return (*this)(i_i, i_j, i_k);
    }

    template <bool C = d2d, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const noexcept {
        std::size_t i_i = i / (dim<1>(sub) / C2);
        std::size_t i_j = i % (dim<1>(sub) / C2);
        return (*this)(i_i, i_j);
    }

    template <bool C = d2d, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const noexcept {
        std::size_t i_i  = i / ((dim<1>(sub) / C1) * (dim<2>(sub) / C2));
        std::size_t i_ij = i % ((dim<1>(sub) / C1) * (dim<2>(sub) / C2));
        std::size_t i_j  = i_ij / (dim<2>(sub) / C2);
        std::size_t i_k  = i_ij % (dim<2>(sub) / C2);

        return (*this)(i_i, i_j, i_k);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(i * C1, j * C2));
    }

    value_type operator()(std::size_t k, std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(k, i * C1, j * C2));
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
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::sum_r_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::mean_r_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t /*unused*/) {
        return etl::dim<0>(v.sub);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::sum_l_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::mean_l_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return etl::size(v.sub) / etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl::dim(v.sub, d + 1);
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

template <typename T, std::size_t C1, std::size_t C2>
struct etl_traits<p_max_pool_p_transformer<T, C1, C2>> {
    using expr_t     = p_max_pool_p_transformer<T, C1, C2>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return etl_traits<sub_expr_t>::size(v.sub) / (C1 * C2);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == dimensions() - 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C2;
        } else if (d == dimensions() - 2) {
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C1;
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, d);
        }
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / (C1 * C2);
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return D == dimensions() - 1 ? etl_traits<sub_expr_t>::template dim<D>() / C2
                                     : D == dimensions() - 2 ? etl_traits<sub_expr_t>::template dim<D>() / C1
                                                             : etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for flipping transformers
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::hflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::vflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::fflip_transformer, std::decay_t<T>>,
                         is_3<etl::p_max_pool_h_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename T::sub_type>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};


/*!
 * \brief Specialization for tranpose_transformer
 */
template <typename T>
struct etl_traits<transpose_transformer<T>> {
    using expr_t     = etl::transpose_transformer<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, 1 - d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<1 - D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for mm_mul_transformer
 */
template <typename LE, typename RE>
struct etl_traits<mm_mul_transformer<LE, RE>> {
    using expr_t       = etl::mm_mul_transformer<LE, RE>;
    using left_expr_t  = std::decay_t<LE>;
    using right_expr_t = std::decay_t<RE>;

    static constexpr const bool is_etl         = true;
    static constexpr const bool is_transformer = true;
    static constexpr const bool is_view        = false;
    static constexpr const bool is_magic_view  = false;
    static constexpr const bool is_fast        = etl_traits<left_expr_t>::is_fast && etl_traits<right_expr_t>::is_fast;
    static constexpr const bool is_linear      = false;
    static constexpr const bool is_value       = false;
    static constexpr const bool is_generator   = false;
    static constexpr const bool vectorizable = false;
    static constexpr const bool needs_temporary_visitor =
        etl_traits<left_expr_t>::needs_temporary_visitor || etl_traits<right_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor =
        etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return dim(v, 0) * dim(v, 1);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return etl::dim(v.left, 0);
        } else {
            cpp_assert(d == 1, "Only 2D mmul are supported");

            return etl::dim(v.right, 1);
        }
    }

    static constexpr std::size_t size() {
        return etl_traits<left_expr_t>::template dim<0>() * etl_traits<right_expr_t>::template dim<1>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(D < 2, "Only 2D mmul are supported");

        return D == 0 ? etl_traits<left_expr_t>::template dim<0>() : etl_traits<right_expr_t>::template dim<1>();
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx_transformer<E>> {
    using expr_t     = etl::dyn_convmtx_transformer<E>;
    using sub_expr_t = std::decay_t<E>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.h * (etl::size(v.sub) + v.h - 1);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return v.h;
        } else {
            return etl::size(v.sub) + v.h - 1;
        }
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx2_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx2_transformer<E>> {
    using expr_t     = etl::dyn_convmtx2_transformer<E>;
    using sub_expr_t = std::decay_t<E>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        auto c_height = (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        auto c_width  = v.k1 * v.k2;
        return c_height * c_width;
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        } else {
            return v.k1 * v.k2;
        }
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for rep_r_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_r_transformer<T, D...>> {
    using expr_t     = etl::rep_r_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr const std::size_t sub_d = etl_traits<sub_expr_t>::dimensions();

    static std::size_t size(const expr_t& v) {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == 0 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0, 0, D...>::value;
    }

    static constexpr std::size_t size() {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D2, cpp_enable_if(D2 < sub_d)>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D2>();
    }

    template <std::size_t D2, cpp_disable_if(D2 < sub_d)>
    static constexpr std::size_t dim() {
        return nth_size<D2 - sub_d, 0, D...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for rep_l_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_l_transformer<T, D...>> {
    using expr_t     = etl::rep_l_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
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
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == dimensions() - 1 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0, 0, D...>::value;
    }

    static constexpr std::size_t size() {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D2>
    static constexpr std::size_t dim() {
        return D2 >= sizeof...(D) ? etl_traits<sub_expr_t>::template dim<D2 - sizeof...(D)>() : nth_size<D2, 0, D...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dyn_rep_r_transformer
 */
template <typename T, std::size_t D>
struct etl_traits<dyn_rep_r_transformer<T, D>> {
    using expr_t     = etl::dyn_rep_r_transformer<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr const std::size_t sub_d = etl_traits<sub_expr_t>::dimensions();

    static std::size_t size(const expr_t& v) {
        return v.m * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d < sub_d ? etl_traits<sub_expr_t>::dim(v.sub, d) : v.reps[d - sub_d];
    }

    static constexpr std::size_t dimensions() {
        return D + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dyn_rep_l_transformer
 */
template <typename T, std::size_t D>
struct etl_traits<dyn_rep_l_transformer<T, D>> {
    using expr_t     = etl::dyn_rep_l_transformer<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = true;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_linear               = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.m * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d >= D ? etl_traits<sub_expr_t>::dim(v.sub, d - D) : v.reps[d];
    }

    static constexpr std::size_t dimensions() {
        return D + etl_traits<sub_expr_t>::dimensions();
    }
};

template <typename T, std::size_t... D>
std::ostream& operator<<(std::ostream& os, const rep_r_transformer<T, D...>& transformer) {
    return os << "rep_r[" << concat_sizes(D...) << "](" << transformer.sub << ")";
}

template <typename T, std::size_t... D>
std::ostream& operator<<(std::ostream& os, const rep_l_transformer<T, D...>& transformer) {
    return os << "rep_l[" << concat_sizes(D...) << "](" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const sum_r_transformer<T>& transformer) {
    return os << "sum_r(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const mean_r_transformer<T>& transformer) {
    return os << "mean_r(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const sum_l_transformer<T>& transformer) {
    return os << "sum_l(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const mean_l_transformer<T>& transformer) {
    return os << "mean_l(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const hflip_transformer<T>& transformer) {
    return os << "hflip(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const vflip_transformer<T>& transformer) {
    return os << "vflip(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const fflip_transformer<T>& transformer) {
    return os << "fflip(" << transformer.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const transpose_transformer<T>& transformer) {
    return os << "T(" << transformer.sub << ")";
}

template <typename L, typename R>
std::ostream& operator<<(std::ostream& os, const mm_mul_transformer<L, R>& transformer) {
    return os << "mm_mul(" << transformer.left << "," << transformer.right << ")";
}

template <typename T, std::size_t C1, std::size_t C2>
std::ostream& operator<<(std::ostream& os, const p_max_pool_h_transformer<T, C1, C2>& transformer) {
    return os << "p_mp_h[" << concat_sizes(C1, C2) << "](" << transformer.sub << ")";
}

template <typename T, std::size_t C1, std::size_t C2>
std::ostream& operator<<(std::ostream& os, const p_max_pool_p_transformer<T, C1, C2>& transformer) {
    return os << "p_mp_p[" << concat_sizes(C1, C2) << "](" << transformer.sub << ")";
}

} //end of namespace etl
