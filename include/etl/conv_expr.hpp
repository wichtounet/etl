//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONVOLUTION_EXPR_HPP
#define ETL_CONVOLUTION_EXPR_HPP

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "traits_lite.hpp"

//Get the implementations
#include "impl/conv.hpp"

namespace etl {

namespace detail {

template<conv_type TT, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    if(TT == conv_type::FULL){
        cpp_assert(dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1, "Invalid sizes for 'full' convolution");
    } else if(TT == conv_type::SAME){
        cpp_assert(dim<0>(conv) == dim<0>(input), "Invalid sizes for 'same' convolution");
    } else if(TT == conv_type::VALID){
        cpp_assert(dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1, "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<conv_type TT, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I& /*input*/, const K& /*kernel*/, const C& /*conv*/){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    static_assert(
        TT != conv_type::FULL || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1,
        "Invalid sizes for 'full'convolution");
    static_assert(
        TT != conv_type::SAME || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>(),
        "Invalid sizes for 'same'convolution");
    static_assert(
        TT != conv_type::VALID || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1,
        "Invalid sizes for 'valid'convolution");
}

template<conv_type TT, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    if(TT == conv_type::FULL){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1
            &&  dim<1>(conv) == dim<1>(input) + dim<1>(kernel) - 1,
            "Invalid sizes for 'full' convolution");
    } else if(TT == conv_type::SAME){
        cpp_assert(dim<0>(conv) == dim<0>(input) && dim<1>(conv) == dim<1>(input), "Invalid sizes for 'same' convolution");
    } else if(TT == conv_type::VALID){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1
            &&  dim<1>(conv) == dim<1>(input) - dim<1>(kernel) + 1,
            "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<conv_type TT, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I& /*input*/, const K& /*kernel*/, const C& /*conv*/){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    static_assert(
        TT != conv_type::FULL || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() + etl_traits<K>::template dim<1>() - 1),
        "Invalid sizes for 'full'convolution");
    static_assert(
        TT != conv_type::SAME || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>()
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>()),
        "Invalid sizes for 'same'convolution");
    static_assert(
        TT != conv_type::VALID || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() - etl_traits<K>::template dim<1>() + 1),
        "Invalid sizes for 'valid'convolution");
}

template<conv_type TT, typename I, typename K, typename C, cpp_enable_if(etl_traits<I>::dimensions() == 3 && etl_traits<I>::is_fast && etl_traits<K>::is_fast && etl_traits<C>::is_fast)>
void check_conv_deep_sizes(const I& i, const K& k, const C& c){
    static_assert(etl_traits<I>::dimensions() == 3 && etl_traits<K>::dimensions() == 3 && etl_traits<C>::dimensions() == 3, "Invalid dimensions for 3D convolution");

    static_assert(
            decay_traits<I>::template dim<0>() == decay_traits<K>::template dim<0>()
        &&  decay_traits<K>::template dim<0>() == decay_traits<C>::template dim<0>(),
        "Deep convolution parameters need to have the same first dimension");

    detail::check_conv_2d_sizes<TT>(i(0), k(0), c(0));
}

template<conv_type TT, typename I, typename K, typename C, cpp_enable_if((etl_traits<I>::dimensions() > 3) && etl_traits<I>::is_fast && etl_traits<K>::is_fast && etl_traits<C>::is_fast)>
void check_conv_deep_sizes(const I& i, const K& k, const C& c){
    static_assert(etl_traits<I>::dimensions() == etl_traits<K>::dimensions() && etl_traits<K>::dimensions() == etl_traits<C>::dimensions(), "Invalid dimensions for 3D convolution");

    static_assert(
            decay_traits<I>::template dim<0>() == decay_traits<K>::template dim<0>()
        &&  decay_traits<K>::template dim<0>() == decay_traits<C>::template dim<0>(),
        "Deep convolution parameters need to have the same first dimension");

    detail::check_conv_deep_sizes<TT>(i(0), k(0), c(0));
}

template<conv_type TT, typename I, typename K, typename C, cpp_enable_if(etl_traits<I>::dimensions() == 3 && (!etl_traits<I>::is_fast || !etl_traits<K>::is_fast || !etl_traits<C>::is_fast))>
void check_conv_deep_sizes(const I& i, const K& k, const C& c){
    static_assert(etl_traits<I>::dimensions() == 3 && etl_traits<K>::dimensions() == 3 && etl_traits<C>::dimensions() == 3, "Invalid dimensions for 3D convolution");

    cpp_assert(
            decay_traits<I>::dim(i, 0) == decay_traits<K>::dim(k, 0)
        &&  decay_traits<K>::dim(k, 0) == decay_traits<C>::dim(c, 0),
        "Deep convolution parameters need to have the same first dimension");

    detail::check_conv_2d_sizes<TT>(i(0), k(0), c(0));
}

template<conv_type TT, typename I, typename K, typename C, cpp_enable_if((etl_traits<I>::dimensions() > 3) && (!etl_traits<I>::is_fast || !etl_traits<K>::is_fast || !etl_traits<C>::is_fast))>
void check_conv_deep_sizes(const I& i, const K& k, const C& c){
    static_assert(etl_traits<I>::dimensions() == etl_traits<K>::dimensions() && etl_traits<K>::dimensions() == etl_traits<C>::dimensions(), "Invalid dimensions for 3D convolution");

    cpp_assert(
            decay_traits<I>::dim(i, 0) == decay_traits<K>::dim(k, 0)
        &&  decay_traits<K>::dim(k, 0) == decay_traits<C>::dim(c, 0),
        "Deep convolution parameters need to have the same first dimension");

    detail::check_conv_deep_sizes<TT>(i(0), k(0), c(0));
}

} //end of namespace detail

template<typename T, std::size_t D, conv_type TT, template<typename...> class Impl>
struct basic_conv_expr {
    using this_type = basic_conv_expr<T, D, TT, Impl>;

    template<typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim(){
        return
                (D > 2 && DD < (D - 2)) ? decay_traits<A>::template dim<DD>()
            :   TT == conv_type::VALID  ? decay_traits<A>::template dim<DD>() - decay_traits<B>::template dim<DD>() + 1
            :   TT == conv_type::SAME   ? decay_traits<A>::template dim<DD>()
            :                             decay_traits<A>::template dim<DD>() + decay_traits<B>::template dim<DD>() - 1;
    }

    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_matrix<value_t<A>, D>;
    };

    template<typename A, typename B, typename I>
    struct fast_result_type_builder;

    template<typename A, typename B, std::size_t... I>
    struct fast_result_type_builder<A, B, std::index_sequence<I...>> {
        using type = fast_dyn_matrix<typename std::decay_t<A>::value_type, this_type::template dim<A,B,I>()...>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<decay_traits<A>::is_fast && decay_traits<B>::is_fast>> {
        using type = typename fast_result_type_builder<A, B, std::make_index_sequence<D>>::type;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&& /*a*/, B&& /*b*/){
        return new result_type<A, B>();
    }

    template<typename A, typename B, std::size_t... I>
    static result_type<A,B>* dyn_allocate(const A& a, const B& b, std::index_sequence<I...> /*seq*/){
        return new result_type<A, B>(this_type::dim(a, b, I)...);
    }

    template<typename A, typename B, cpp_disable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&& a, B&& b){
        return dyn_allocate(std::forward<A>(a), std::forward<B>(b), std::make_index_sequence<D>());
    }

    template<typename A, typename B, typename C, std::size_t D2 = D, cpp_enable_if(D2 == 1)>
    static void check(const A& a, const B& b, const C& c){
        detail::check_conv_1d_sizes<TT>(a, b, c);
    }

    template<typename A, typename B, typename C, std::size_t D2 = D, cpp_enable_if(D2 == 2)>
    static void check(const A& a, const B& b, const C& c){
        detail::check_conv_2d_sizes<TT>(a, b, c);
    }

    template<typename A, typename B, typename C, std::size_t D2 = D, cpp_enable_if((D2 > 2))>
    static void check(const A& a, const B& b, const C& c){
        detail::check_conv_deep_sizes<TT>(a, b, c);
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

        check(a, b, c);

        if(D == 1 || D == 2){
            Impl<decltype(make_temporary(std::forward<A>(a))), decltype(make_temporary(std::forward<B>(b))), C, void>::apply(
                make_temporary(std::forward<A>(a)),
                make_temporary(std::forward<B>(b)),
                std::forward<C>(c));
        } else {
            Impl<A,B,C,void>::apply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }

    static std::string desc() noexcept {
        if(TT == conv_type::VALID){
            return "conv_valid";
        } else if(TT == conv_type::SAME){
            return "conv_same";
        } else {
            return "conv_full";
        }
    }

    template<typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d){
        if(D > 2 && d < (D - 2)){
            return etl_traits<A>::dim(a, d);
        } else {
            if(TT == conv_type::VALID){
                return etl_traits<A>::dim(a, d) - etl_traits<B>::dim(b, d) + 1;
            } else if(TT == conv_type::SAME){
                return etl_traits<A>::dim(a, d);
            } else {
                return etl_traits<A>::dim(a, d) + etl_traits<B>::dim(b, d) - 1;
            }
        }
    }

    template<typename A, typename B>
    static std::size_t size(const A& a, const B& b){
        if(D > 2){
            std::size_t acc = 1;
            for(std::size_t i = 0; i < D; ++i){
                acc *= this_type::dim(a, b, i);
            }
            return acc;
        } else if(D == 1){
            return this_type::dim(a, b, 0);
        } else { //D == 2
            return this_type::dim(a, b, 0) * this_type::dim(a, b, 1);
        }
    }

    template<typename A, typename B, std::size_t... I>
    static constexpr std::size_t size_mul(const std::index_sequence<I...>& ){
        return mul_all<this_type::dim<A, B, I>()...>::value;
    }

    template<typename A, typename B>
    static constexpr std::size_t size(){
        return size_mul<A, B>(std::make_index_sequence<D>());
    }

    static constexpr std::size_t dimensions(){
        return D;
    }
};

//1D convolution

template<typename T>
using conv1_valid_expr = basic_conv_expr<T, 1, conv_type::VALID, detail::conv1_valid_impl>;

template<typename T>
using conv1_same_expr = basic_conv_expr<T, 1, conv_type::SAME, detail::conv1_same_impl>;

template<typename T>
using conv1_full_expr = basic_conv_expr<T, 1, conv_type::FULL, detail::conv1_full_impl>;

template<typename T>
using fft_conv1_full_expr = basic_conv_expr<T, 1, conv_type::FULL, detail::fft_conv1_full_impl>;

//2D convolutions

template<typename T>
using conv2_valid_expr = basic_conv_expr<T, 2, conv_type::VALID, detail::conv2_valid_impl>;

template<typename T>
using conv2_same_expr = basic_conv_expr<T, 2, conv_type::SAME, detail::conv2_same_impl>;

template<typename T>
using conv2_full_expr = basic_conv_expr<T, 2, conv_type::FULL, detail::conv2_full_impl>;

template<typename T>
using fft_conv2_full_expr = basic_conv_expr<T, 2, conv_type::FULL, detail::fft_conv2_full_impl>;

//>2D convolutions

template<typename T, std::size_t D>
using conv_deep_valid_expr = basic_conv_expr<T, D, conv_type::VALID, detail::conv_deep_valid_impl>;

template<typename T, std::size_t D>
using conv_deep_same_expr = basic_conv_expr<T, D, conv_type::SAME, detail::conv_deep_same_impl>;

template<typename T, std::size_t D>
using conv_deep_full_expr = basic_conv_expr<T, D, conv_type::FULL, detail::conv_deep_full_impl>;

//Deep convolutions

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        conv(i)  = conv_2d_full(input(i), kernel(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_full(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        conv(i) = conv_2d_same(input(i), kernel(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_same(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        conv(i) = conv_2d_valid(input(i), kernel(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_valid(input(i), kernel(i), conv(i));
    }

    return conv;
}

} //end of namespace etl

#endif
