//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRANSFORMERS_HPP
#define ETL_TRANSFORMERS_HPP

#include "tmp.hpp"
#include "traits_fwd.hpp"

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

//max pool is not really a transformer, but the implemented version needs
//access to the position to be computed and therefore does not provide an
//operator[] which makes it unstable.

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

    explicit p_max_pool_h_transformer(sub_type vec) : base_type(vec) {}

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

    explicit p_max_pool_p_transformer(sub_type vec) : base_type(vec) {}

    value_type operator()(std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(i * C1, j * C2));
    }

    value_type operator()(std::size_t k, std::size_t i, std::size_t j) const {
        return 1.0 / (1.0 + base_type::pool(k, i * C1, j * C2));
    }
};

} //end of namespace etl

#endif
