//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template <template <typename, std::size_t> class TT, typename T>
struct is_2 : std::false_type {};

template <template <typename, std::size_t> class TT, typename V1, std::size_t R>
struct is_2<TT, TT<V1, R>> : std::true_type {};

template <template <typename, std::size_t, std::size_t> class TT, typename T>
struct is_3 : std::false_type {};

template <template <typename, std::size_t, std::size_t> class TT, typename V1, std::size_t R1, std::size_t R2>
struct is_3<TT, TT<V1, R1, R2>> : std::true_type {};

template <template <typename, std::size_t...> class TT, typename T>
struct is_var : std::false_type {};

template <template <typename, std::size_t...> class TT, typename V1, std::size_t... R>
struct is_var<TT, TT<V1, R...>> : std::true_type {};

template <template <typename, typename, std::size_t...> class TT, typename T>
struct is_var_2 : std::false_type {};

template <template <typename, typename, std::size_t...> class TT, typename V1, typename V2, std::size_t... R>
struct is_var_2<TT, TT<V1, V2, R...>> : std::true_type {};

template <typename E>
using value_t = typename std::decay_t<E>::value_type;

template <typename S>
using memory_t = std::conditional_t<
    std::is_const<std::remove_reference_t<S>>::value,
    typename std::decay_t<S>::const_memory_type,
    typename std::decay_t<S>::memory_type>;

template <std::size_t F, std::size_t... Dims>
struct mul_all final : std::integral_constant<std::size_t, F * mul_all<Dims...>::value> {};

template <std::size_t F>
struct mul_all<F> final : std::integral_constant<std::size_t, F> {};

template <std::size_t S, std::size_t I, std::size_t F, std::size_t... Dims>
struct nth_size final {
    template <std::size_t S2, std::size_t I2, typename Enable = void>
    struct nth_size_int : std::integral_constant<std::size_t, nth_size<S, I + 1, Dims...>::value> {};

    template <std::size_t S2, std::size_t I2>
    struct nth_size_int<S2, I2, std::enable_if_t<S2 == I2>> : std::integral_constant<std::size_t, F> {};

    static constexpr const std::size_t value = nth_size_int<S, I>::value;
};

template <std::size_t... D, cpp_enable_if(sizeof...(D) == 0)>
std::size_t dyn_nth_size(std::size_t /*d*/) {
    cpp_assert(false, "Should never be called");
    return 0;
}

template <std::size_t D1, std::size_t... D>
std::size_t dyn_nth_size(std::size_t i) {
    return i == 0
               ? D1
               : dyn_nth_size<D...>(i - 1)
}

template <typename S1, typename S2, typename Enable = void>
struct sequence_equal;

template <>
struct sequence_equal<std::index_sequence<>, std::index_sequence<>> : std::true_type {};

template <std::size_t... I1, std::size_t... I2>
struct sequence_equal<std::index_sequence<I1...>, std::index_sequence<I2...>,
                      std::enable_if_t<sizeof...(I1) != sizeof...(I2)>> : std::false_type {};

template <std::size_t I, std::size_t... I1, std::size_t... I2>
struct sequence_equal<std::index_sequence<I, I1...>, std::index_sequence<I, I2...>,
                      std::enable_if_t<sizeof...(I1) == sizeof...(I2)>> : sequence_equal<std::index_sequence<I1...>, std::index_sequence<I2...>> {};

template <std::size_t I11, std::size_t I21, std::size_t... I1, std::size_t... I2>
struct sequence_equal<std::index_sequence<I11, I1...>, std::index_sequence<I21, I2...>,
                      cpp::disable_if_t<I11 == I21>> : std::false_type {};

template <typename Int, typename, Int Begin>
struct integer_range_impl;

template <typename Int, Int... N, Int Begin>
struct integer_range_impl<Int, std::integer_sequence<Int, N...>, Begin> {
    using type = std::integer_sequence<Int, N + Begin...>;
};

template <typename Int, Int Begin, Int End>
using make_integer_range = typename integer_range_impl<Int, std::make_integer_sequence<Int, End - Begin>, Begin>::type;

template <std::size_t Begin, std::size_t End>
using make_index_range = make_integer_range<std::size_t, Begin, End>;

template <typename... Dims>
std::string concat_sizes(Dims... sizes) {
    std::array<std::size_t, sizeof...(Dims)> tmp{{sizes...}};
    std::string result;
    std::string sep;
    for (auto& v : tmp) {
        result += sep;
        result += std::to_string(v);
        sep = ",";
    }
    return result;
}

struct dereference_op {
    template <typename T>
    static decltype(auto) apply(T&& t) {
        return *(std::forward<T>(t));
    }
};

struct forward_op {
    template <typename T>
    static decltype(auto) apply(T&& t) {
        return std::forward<T>(t);
    }
};

template <bool B, typename T, cpp_enable_if(B)>
constexpr decltype(auto) optional_move(T&& t) {
    return std::move(t);
}

template <bool B, typename T, cpp_disable_if(B)>
constexpr decltype(auto) optional_move(T&& t) {
    return std::forward<T>(t);
}

} //end of namespace etl
