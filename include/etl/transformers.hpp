//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRANSFORMERS_HPP
#define ETL_TRANSFORMERS_HPP

#include <iostream> //For stream support

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "tmp.hpp"
#include "traits_lite.hpp"

namespace etl {

template<typename T, std::size_t... D>
struct rep_r_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit rep_r_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sub(i / mul_all<D...>::value);
    }

    template<typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return sub(i);
    }
};

template<typename T, std::size_t... D>
struct rep_l_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit rep_l_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sub(i % size(sub));
    }

    template<typename... Sizes>
    value_type operator()(Sizes... sizes) const {
        return sub(cpp::last_value(sizes...));
    }
};

template<typename T>
struct sum_r_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit sum_r_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sum(sub(i));
    }

    template<typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return sum(sub(i));
    }
};

template<typename T>
struct mean_r_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit mean_r_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return mean(sub(i));
    }

    template<typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return mean(sub(i));
    }
};

template<typename T>
struct sum_l_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit sum_l_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sum(col(sub,i));
    }

    value_type operator()(std::size_t i) const {
        return sum(col(sub,i));
    }
};

template<typename T>
struct mean_l_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit mean_l_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t j) const {
        value_type m = 0.0;

        for(std::size_t i = 0; i < dim<0>(sub); ++i){
            m += sub[j + i * (size(sub) / dim<0>(sub))];
        }

        return m / dim<0>(sub);
    }

    template<typename... Sizes>
    value_type operator()(std::size_t j, Sizes... sizes) const {
        value_type m = 0.0;

        for(std::size_t i = 0; i < dim<0>(sub); ++i){
            m += sub(i, j, sizes...);
        }

        return m / dim<0>(sub);
    }
};

template<typename T>
struct hflip_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit hflip_transformer(sub_type vec) : sub(vec) {}

    static constexpr const auto matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    template<bool C = matrix, cpp::disable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        return sub[size(sub) - i - 1];
    }

    template<bool C = matrix, cpp::enable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / dim<1>(sub);
        auto i_j = i % dim<1>(sub);
        return sub[i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j)];
    }

    value_type operator()(std::size_t i) const {
        return sub(size(sub) - 1 - i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(i, columns(sub) - 1 - j);
    }
};

template<typename T>
struct vflip_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit vflip_transformer(sub_type vec) : sub(vec) {}

    static constexpr const auto matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    template<bool C = matrix, cpp::disable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        return sub[i];
    }

    template<bool C = matrix, cpp::enable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / dim<1>(sub);
        auto i_j = i % dim<1>(sub);
        return sub[(dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j];
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, j);
    }
};

template<typename T>
struct fflip_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit fflip_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        if(dimensions(sub) == 1){
            return sub[i];
        } else {
            return sub[size(sub) - i - 1];
        }
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, columns(sub) - 1 - j);
    }
};

template<typename T>
struct transpose_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit transpose_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        if(dimensions(sub) == 1){
            return sub[i];
        } else {
            return sub[(i % dim<0>(sub)) * dim<1>(sub) + (i / dim<0>(sub))];
        }
    }

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(j, i);
    }
};

template<typename L, typename R>
struct mm_mul_transformer {
    using left_type = L;
    using right_type = R;
    using value_type = value_t<L>;

    left_type left;
    right_type right;

    template<typename A, typename B, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast> = cpp::detail::dummy>
    void check_mmul_sizes(const A& a, const B& b){
        cpp_assert(
                dim<1>(a) == dim<0>(b)          //interior dimensions
            , "Invalid sizes for multiplication");
        cpp_unused(a);
        cpp_unused(b);
    }

    template<typename A, typename B, cpp::enable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast> = cpp::detail::dummy>
    void check_mmul_sizes(const A&, const B&){
        static_assert(
                etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()          //interior dimensions
            , "Invalid sizes for multiplication");
    }

    mm_mul_transformer(left_type left, right_type right) : left(left), right(right) {
        check_mmul_sizes(left, right);
    }

    value_type operator[](std::size_t i) const {
        auto i_i = i / dim<1>(right);
        auto i_j = i % dim<1>(right);
        return (*this)(i_i, i_j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        value_type c = 0;

        for(std::size_t k = 0; k < columns(left); k++){
            c += left(i,k) * right(k,j);
        }

        return c;
    }
};

template<typename T>
struct dyn_convmtx_transformer {
    using sub_type = T;
    using value_type = value_t<sub_type>;

    static_assert(decay_traits<T>::dimensions() == 1, "convmtx can only be applied on vectors");

    sub_type sub;
    std::size_t h;

    dyn_convmtx_transformer(sub_type sub, std::size_t h) : sub(sub), h(h) {}

    value_type operator[](std::size_t i) const {
        auto i_i = i / (etl::size(sub) + h - 1);
        auto i_j = i % (etl::size(sub) + h - 1);
        return (*this)(i_i, i_j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        if(j < i){
            return value_type(0);
        } else if(j >= etl::size(sub) + i){
            return value_type(0);
        } else {
            return sub(j - i);
        }
    }
};

template<typename T>
struct dyn_convmtx2_transformer {
    using sub_type = T;
    using value_type = value_t<sub_type>;

    static_assert(decay_traits<T>::dimensions() == 2, "convmtx2 can only be applied on matrices");

    sub_type sub;
    const std::size_t k1;
    const std::size_t k2;
    std::size_t i1;
    std::size_t i2;
    std::size_t inner_paddings;
    std::size_t inner_padding;

    dyn_convmtx2_transformer(sub_type sub, std::size_t k1, std::size_t k2) : sub(sub), k1(k1), k2(k2) {
        i1 = etl::dim<0>(sub);
        i2 = etl::dim<1>(sub);

        auto c_height = (i1 + k1 - 1) * (i2 + k2 - 1);
        auto c_width = k1 * k2;

        auto max_fill = c_height - ((i1 + k1 - 1) * ((c_width - 1) / k1) + (c_width - 1) % k1);
        inner_paddings = max_fill - (i1 * i2);
        inner_padding = inner_paddings / (i2 - 1);
    }

    value_type operator[](std::size_t i) const {
        auto i_i = i / (k1 * k2);
        auto i_j = i % (k1 * k2);
        return (*this)(i_i, i_j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        auto top_padding = (i1 + k1 - 1) * (j / k1) + j % k1;

        if(i < top_padding || i >= top_padding + (i1 * i2) + inner_paddings){
            return value_type(0);
        } else {
            auto inner = i - top_padding;
            auto col = inner % (i1 + inner_padding);
            auto block = inner / (i1 + inner_padding);

            if(col >= i1){
                return value_type(0);
            } else {
                return sub(col, block);
            }
        }
    }
};

template<typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit p_max_pool_transformer(sub_type vec) : sub(vec) {}

    value_type pool(std::size_t i, std::size_t j) const {
        value_type p = 0;

        auto start_ii = (i / C1) * C1;
        auto start_jj = (j / C2) * C2;

        for(std::size_t ii = start_ii; ii < start_ii + C1; ++ii){
            for(std::size_t jj  = start_jj; jj < start_jj + C2; ++jj){
                p += std::exp(sub(ii, jj));
            }
        }

        return p;
    }

    value_type pool(std::size_t k, std::size_t i, std::size_t j) const {
        value_type p = 0;

        auto start_ii = (i / C1) * C1;
        auto start_jj = (j / C2) * C2;

        for(std::size_t ii = start_ii; ii < start_ii + C1; ++ii){
            for(std::size_t jj  = start_jj; jj < start_jj + C2; ++jj){
                p += std::exp(sub(k, ii, jj));
            }
        }

        return p;
    }
};

template<typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_h_transformer : p_max_pool_transformer<T, C1, C2> {
    using base_type = p_max_pool_transformer<T, C1, C2>;
    using sub_type = typename base_type::sub_type;
    using value_type = typename base_type::value_type;

    using base_type::sub;

    static constexpr const auto d2d = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    explicit p_max_pool_h_transformer(sub_type vec) : base_type(vec) {}

    template<bool C = d2d, cpp::enable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / dim<1>(sub);
        auto i_j = i % dim<1>(sub);
        return (*this)(i_i, i_j);
    }

    template<bool C = d2d, cpp::disable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / (dim<1>(sub) * dim<2>(sub));
        auto i_ij = i % (dim<1>(sub) * dim<2>(sub));
        auto i_j = i_ij / dim<2>(sub);
        auto i_k = i_ij % dim<2>(sub);

        return (*this)(i_i, i_j, i_k);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return std::exp(sub(i, j)) / (1.0 + base_type::pool(i, j));
    }

    value_type operator()(std::size_t k, std::size_t i, std::size_t j) const {
        return std::exp(sub(k, i, j)) / (1.0 + base_type::pool(k, i, j));
    }
};

template<typename T, std::size_t C1, std::size_t C2>
struct p_max_pool_p_transformer : p_max_pool_transformer<T, C1, C2> {
    using base_type = p_max_pool_transformer<T, C1, C2>;
    using sub_type = typename base_type::sub_type;
    using value_type = typename base_type::value_type;

    using base_type::sub;

    static constexpr const auto d2d = etl_traits<std::decay_t<sub_type>>::dimensions() == 2;

    explicit p_max_pool_p_transformer(sub_type vec) : base_type(vec) {}

    template<bool C = d2d, cpp::enable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / (dim<1>(sub) / C2);
        auto i_j = i % (dim<1>(sub)/ C2);
        return (*this)(i_i, i_j);
    }

    template<bool C = d2d, cpp::disable_if_u<C> = cpp::detail::dummy>
    value_type operator[](std::size_t i) const {
        auto i_i = i / ((dim<1>(sub) / C1) * (dim<2>(sub) / C2));
        auto i_ij = i % ((dim<1>(sub) / C1)* (dim<2>(sub) / C2));
        auto i_j = i_ij / (dim<2>(sub) / C2);
        auto i_k = i_ij % (dim<2>(sub) / C2);

        return (*this)(i_i, i_j, i_k);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(i * C1, j * C2));
    }

    value_type operator()(std::size_t k, std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(k, i * C1, j * C2));
    }
};

template<typename T, std::size_t... D>
std::ostream& operator<<(std::ostream& os, const rep_r_transformer<T, D...>& transformer){
    return os << "rep_r[" << concat_sizes(D...) << "](" << transformer.sub << ")";
}

template<typename T, std::size_t... D>
std::ostream& operator<<(std::ostream& os, const rep_l_transformer<T, D...>& transformer){
    return os << "rep_l[" << concat_sizes(D...) << "](" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const sum_r_transformer<T>& transformer){
    return os << "sum_r(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const mean_r_transformer<T>& transformer){
    return os << "mean_r(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const sum_l_transformer<T>& transformer){
    return os << "sum_l(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const mean_l_transformer<T>& transformer){
    return os << "mean_l(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const hflip_transformer<T>& transformer){
    return os << "hflip(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const vflip_transformer<T>& transformer){
    return os << "vflip(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const fflip_transformer<T>& transformer){
    return os << "fflip(" << transformer.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const transpose_transformer<T>& transformer){
    return os << "T(" << transformer.sub << ")";
}

template<typename L, typename R>
std::ostream& operator<<(std::ostream& os, const mm_mul_transformer<L,R>& transformer){
    return os << "mm_mul(" << transformer.left << "," << transformer.right << ")";
}

template<typename T, std::size_t C1, std::size_t C2>
std::ostream& operator<<(std::ostream& os, const p_max_pool_h_transformer<T, C1, C2>& transformer){
    return os << "p_mp_h[" << concat_sizes(C1,C2) << "](" << transformer.sub << ")";
}

template<typename T, std::size_t C1, std::size_t C2>
std::ostream& operator<<(std::ostream& os, const p_max_pool_p_transformer<T, C1, C2>& transformer){
    return os << "p_mp_p[" << concat_sizes(C1,C2) << "](" << transformer.sub << ")";
}

} //end of namespace etl

#endif
