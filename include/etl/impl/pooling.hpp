//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

template <typename A, typename M>
struct max_pool_2d {
    template <std::size_t C1, std::size_t C2>
    static auto pool_block(const A& sub, std::size_t j, std::size_t k){
        auto max = sub(j * C1, k * C2);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                max = std::max(max, sub(j * C1 + jj, k * C2 + kk));
            }
        }

        return max;
    }

    template <std::size_t C1, std::size_t C2>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = etl::dim<0>(sub) / C1;
        const std::size_t o2 = etl::dim<1>(sub) / C2;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block<C1, C2>(sub, j, k);
            }
        }
    }
};

template <typename A, typename M>
struct avg_pool_2d {
    template <std::size_t C1, std::size_t C2>
    static auto pool_block(const A& sub, std::size_t j, std::size_t k){
        value_t<A> avg = 0;

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                avg += sub(j * C1 + jj, k * C2 + kk);
            }
        }

        return avg / (C1 * C2);
    }

    template <std::size_t C1, std::size_t C2>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = etl::dim<0>(sub) / C1;
        const std::size_t o2 = etl::dim<1>(sub) / C2;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block<C1, C2>(sub, j, k);
            }
        }
    }
};

template <typename A, typename M>
struct max_pool_3d {
    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k){
        auto max = sub(i * C1, j * C2, k * C3);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    max = std::max(max, sub(i * C1 + ii, j * C2 + jj, k * C3 + kk));
                }
            }
        }

        return max;
    }

    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = etl::dim<0>(sub) / C1;
        const std::size_t o2 = etl::dim<1>(sub) / C2;
        const std::size_t o3 = etl::dim<2>(sub) / C3;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block<C1, C2, C3>(sub, i, j, k);
                }
            }
        }
    }
};

template <typename A, typename M>
struct avg_pool_3d {
    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k){
        value_t<A> avg = 0;

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    avg += sub(i * C1 + ii, j * C2 + jj, k * C3 + kk);
                }
            }
        }

        return avg / (C1 * C2 * C3);
    }

    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = etl::dim<0>(sub) / C1;
        const std::size_t o2 = etl::dim<1>(sub) / C2;
        const std::size_t o3 = etl::dim<2>(sub) / C3;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block<C1, C2, C3>(sub, i, j, k);
                }
            }
        }
    }
};

template <typename A, typename B, typename M>
struct max_pool_derivative_2d {
    template <std::size_t C1, std::size_t C2>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t j, std::size_t k){
        auto max = out(j, k);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                if (max == in(j * C1 + jj, k * C2 + kk)) {
                    m(j * C1 + jj, k * C2 + kk) = 1.0;
                } else {
                    m(j * C1 + jj, k * C2 + kk) = 0.0;
                }
            }
        }
    }


    template <std::size_t C1, std::size_t C2>
    static void apply(A&& in, B&& out, M& m) {
        for (std::size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block<C1, C2>(in, out, m, j, k);
            }
        }
    }
};

template <typename A, typename B, typename M>
struct max_pool_derivative_3d {
    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t i, std::size_t j, std::size_t k){
        auto max = out(i, j, k);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    if (max == in(i * C1 + ii, j * C2 + jj, k * C3 + kk)) {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 1.0;
                    } else {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 0.0;
                    }
                }
            }
        }
    }

    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void apply(A&& in, B&& out, M& m) {
        for (std::size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block<C1, C2, C3>(in, out, m, i, j, k);
                }
            }
        }
    }
};

template <typename A, typename M>
struct upsample_2d {
    template <std::size_t C1, std::size_t C2>
    static void upsample_block(A&& in, M& m, std::size_t j, std::size_t k) {
        auto value = in(j, k);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                m(j * C1 + jj, k * C2 + kk) = value;
            }
        }
    }

    template <std::size_t C1, std::size_t C2>
    static void apply(A&& in, M& m) {
        for (std::size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block<C1, C2>(in, m, j, k);
            }
        }
    }
};

template <typename A, typename M>
struct upsample_3d {
    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void upsample_block(A&& in, M& m, std::size_t i, std::size_t j, std::size_t k) {
        auto value = in(i, j, k);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = value;
                }
            }
        }
    }

    template <std::size_t C1, std::size_t C2, std::size_t C3>
    static void apply(A&& in, M& m) {
        for (std::size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block<C1, C2, C3>(in, m, i, j, k);
                }
            }
        }
    }
};

} //end of namespace impl

} //end of namespace etl
