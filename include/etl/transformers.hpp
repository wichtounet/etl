//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRANSFORMERS_HPP
#define ETL_TRANSFORMERS_HPP

#include "tmp.hpp"

namespace etl {

template<typename T, std::size_t... D>
struct rep_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit rep_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sub(i / mul_all<D...>::value);
    }

    template<typename... Sizes>
    value_type operator()(std::size_t i, Sizes... /*sizes*/) const {
        return sub(i);
    }
};

template<typename T>
struct sum_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit sum_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return sum(sub(i));
    }

    value_type operator()(std::size_t i) const {
        return sum(sub(i));
    }
};

template<typename T>
struct mean_transformer {
    using sub_type = T;
    using value_type = value_t<T>;

    sub_type sub;

    explicit mean_transformer(sub_type vec) : sub(vec) {}

    value_type operator[](std::size_t i) const {
        return mean(sub(i));
    }

    value_type operator()(std::size_t i) const {
        return mean(sub(i));
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

    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(j, i);
    }
};

} //end of namespace etl

#endif
