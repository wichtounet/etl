//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Traits to test if T is a specialization of TT<T, size_t>
 */
template <template <typename, size_t> typename TT, typename T>
struct is_2 : std::false_type {};

/*!
 * \copydoc is_2
 */
template <template <typename, size_t> typename TT, typename V1, size_t R>
struct is_2<TT, TT<V1, R>> : std::true_type {};

/*!
 * \brief Traits to test if T is a specialization of TT<T, size_t, size_t>
 */
template <template <typename, size_t, size_t> typename TT, typename T>
struct is_3 : std::false_type {};

/*!
 * \copydoc is_3
 */
template <template <typename, size_t, size_t> typename TT, typename V1, size_t R1, size_t R2>
struct is_3<TT, TT<V1, R1, R2>> : std::true_type {};

/*!
 * \brief Traits to test if T is a specialization of TT<T, size_t...>
 */
template <template <typename, size_t...> typename TT, typename T>
struct is_var : std::false_type {};

/*!
 * \copydoc is_var
 */
template <template <typename, size_t...> typename TT, typename V1, size_t... R>
struct is_var<TT, TT<V1, R...>> : std::true_type {};

/*!
 * \brief Traits to test if T is a specialization of TT<T1, T2, size_t...>
 */
template <template <typename, typename, size_t...> typename TT, typename T>
struct is_var_2 : std::false_type {};

/*!
 * \copydoc is_var_2
 */
template <template <typename, typename, size_t...> typename TT, typename V1, typename V2, size_t... R>
struct is_var_2<TT, TT<V1, V2, R...>> : std::true_type {};

/*!
 * \brief Traits to get information about ETL types
 *
 * For non-ETL types, is_etl is false and in that case, no other fields should be used on the traits.
 *
 * \tparam T the type to introspect
 */
template <typename T, typename Enable = void>
struct etl_traits;

/*!
 * \brief Traits helper to get information about ETL types, the type is first decayed.
 * \tparam E the type to introspect
 */
template <typename E>
using decay_traits = etl_traits<std::decay_t<E>>;

/*!
 * \brief Traits to extract the value type out of an ETL type
 */
template <typename E>
using value_t = typename decay_traits<E>::value_type;

/*!
 * \brief Traits to extract the direct memory type out of an ETL type
 */
template <typename S>
using memory_t = std::conditional_t<
    std::is_const<std::remove_reference_t<S>>::value,
    typename std::decay_t<S>::const_memory_type,
    typename std::decay_t<S>::memory_type>;

/*!
 * \brief Traits to extract the direct const memory type out of an ETL type
 */
template <typename S>
using const_memory_t = typename std::decay_t<S>::const_memory_type;

/*!
 * \brief Value traits to compute the multiplication of all the given values
 */
template <size_t F, size_t... Dims>
struct mul_all_impl final : std::integral_constant<size_t, F * mul_all_impl<Dims...>::value> {};

/*!
 * \copydoc mul_all_impl
 */
template <size_t F>
struct mul_all_impl<F> final : std::integral_constant<size_t, F> {};

//CPP17: Can be totally replaced with variadic fold expansion

/*!
 * \brief Value traits to compute the multiplication of all the given values
 */
template <size_t F, size_t... Dims>
constexpr size_t mul_all = mul_all_impl<F, Dims...>::value;

/*!
 * \brief Traits to get the Sth dimension in Dims..
 * \tparam S The searched dimension
 * \tparam I The current index (start at zero)
 */
template <size_t S, size_t I, size_t F, size_t... Dims>
struct nth_size_impl final {
    /*!
     * \brief Helper traits to get the S2th dimension in Dims... (of
     * the parent class)
     * \tparam S2 The searched dimension
     * \tparam I2 The current index (start at zero)
     */
    template <size_t S2, size_t I2, typename Enable = void>
    struct nth_size_int : std::integral_constant<size_t, nth_size_impl<S, I + 1, Dims...>::value> {};

    /*!
     * \brief Helper traits to get the S2th dimension in Dims... (of
     * the parent class)
     * \tparam S2 The searched dimension
     * \tparam I2 The current index (start at zero)
     */
    template <size_t S2, size_t I2>
    struct nth_size_int<S2, I2, std::enable_if_t<S2 == I2>> : std::integral_constant<size_t, F> {};

    static constexpr size_t value = nth_size_int<S, I>::value; ///< The result value
};

/*!
 * \brief Traits to get the Sth dimension in Dims..
 * \tparam S The searched dimension
 * \tparam I The current index (start at zero)
 */
template <size_t S, size_t I, size_t F, size_t... Dims>
constexpr size_t nth_size = nth_size_impl<S, I, F, Dims...>::value;

/*!
 * \brief Returns the dth (dynamic) dimension from the variadic list D
 * \tparam D The list of dimensions
 * \param d The index of the dimension to get
 */
template <size_t... D, cpp_enable_iff(sizeof...(D) == 0)>
size_t dyn_nth_size(size_t d) {
    cpp_unused(d);
    cpp_assert(false, "Should never be called");
    return 0;
}

/*!
 * \copydoc dyn_nth_size
 */
template <size_t D1, size_t... D>
size_t dyn_nth_size(size_t i) {
    return i == 0
               ? D1
               : dyn_nth_size<D...>(i - 1);
}

/*!
 * \brief Traits to test if two index_sequence are equal
 */
template <typename S1, typename S2, typename Enable = void>
struct sequence_equal;

/*!
 * \copydoc sequence_equal
 */
template <>
struct sequence_equal<std::index_sequence<>, std::index_sequence<>> : std::true_type {};

/*!
 * \copydoc sequence_equal
 */
template <size_t... I1, size_t... I2>
struct sequence_equal<std::index_sequence<I1...>, std::index_sequence<I2...>,
                      std::enable_if_t<sizeof...(I1) != sizeof...(I2)>> : std::false_type {};

/*!
 * \copydoc sequence_equal
 */
template <size_t I, size_t... I1, size_t... I2>
struct sequence_equal<std::index_sequence<I, I1...>, std::index_sequence<I, I2...>,
                      std::enable_if_t<sizeof...(I1) == sizeof...(I2)>> : sequence_equal<std::index_sequence<I1...>, std::index_sequence<I2...>> {};

/*!
 * \copydoc sequence_equal
 */
template <size_t I11, size_t I21, size_t... I1, size_t... I2>
struct sequence_equal<std::index_sequence<I11, I1...>, std::index_sequence<I21, I2...>,
                      cpp::disable_if_t<I11 == I21>> : std::false_type {};

/*!
 * \brief Implementation for TMP utility to hold a range of integers
 * \tparam Int The type of the integers
 * \tparam The integer sequence from 0 to End - Begin
 * \tparam Begin The first integer in the sequence
 */
template <typename Int, typename T, Int Begin>
struct integer_range_impl;

/*!
 * \copydoc integer_range_impl
 */
template <typename Int, Int... N, Int Begin>
struct integer_range_impl<Int, std::integer_sequence<Int, N...>, Begin> {
    /*!
     * \brief The resulting integer range from Begin to End
     */
    using type = std::integer_sequence<Int, N + Begin...>;
};

/*!
 * \brief Helper to create an integer_range of numbers
 */
template <typename Int, Int Begin, Int End>
using make_integer_range = typename integer_range_impl<Int, std::make_integer_sequence<Int, End - Begin>, Begin>::type;

/*!
 * \brief Helper to create an integer_range of size_t numbers
 */
template <size_t Begin, size_t End>
using make_index_range = make_integer_range<size_t, Begin, End>;

/*!
 * \brief Returns a string representation of the given dimensions
 */
template <typename... Dims>
std::string concat_sizes(Dims... sizes) {
    std::array<size_t, sizeof...(Dims)> tmp{{sizes...}};
    std::string result;
    std::string sep;
    for (auto& v : tmp) {
        result += sep;
        result += std::to_string(v);
        sep = ",";
    }
    return result;
}

/*!
 * \brief Functor that dereference a pointer and return its value
 */
struct dereference_op {
    /*!
     * \brief Apply the functor on t
     */
    template <typename T>
    static decltype(auto) apply(T&& t) {
        return *(std::forward<T>(t));
    }
};

/*!
 * \brief Functor that forwards a value
 */
struct forward_op {
    /*!
     * \brief Apply the functor on t
     */
    template <typename T>
    static decltype(auto) apply(T&& t) {
        return std::forward<T>(t);
    }
};

template <typename T>
using remove_const_deep =
    std::conditional_t<
        std::is_lvalue_reference<T>::value,
        std::add_lvalue_reference_t<std::remove_const_t<std::remove_reference_t<T>>>,
        std::remove_const_t<T>>;

/*!
 * \brief Functor that forwards a value and removes the constness of
 * it.
 */
struct forward_op_nc {
    /*!
     * \brief Apply the functor on t
     */
    template <typename T>
    static decltype(auto) apply(T&& t) {
        using real_type = decltype(std::forward<T>(t));
        return const_cast<remove_const_deep<real_type>>(std::forward<T>(t));
    }
};

/*!
 * \brief Function to move or forward depending on a constant boolean flag
 * \tparam B Decides if return is moving (true) or forwarding (false)
 */
template <bool B, typename T, cpp_enable_iff(B)>
constexpr decltype(auto) optional_move(T&& t) {
    return std::move(t);
}

/*!
 * \copydoc optional_move
 */
template <bool B, typename T, cpp_disable_iff(B)>
constexpr decltype(auto) optional_move(T&& t) {
    return std::forward<T>(t);
}

} //end of namespace etl
