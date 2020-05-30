//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/bernoulli.hpp"

namespace etl {

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename T>
struct bernoulli_unary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_sbernoulli_sample_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dbernoulli_sample_seed);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X & x, Y& y) noexcept {
        static random_engine rand_engine(std::time(nullptr));

        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1);
        impl::egblas::bernoulli_sample_seed(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, seed_dist(rand_engine));

        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X & x, Y& y) noexcept {
        static random_engine rand_engine(std::time(nullptr));

        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1);
        impl::egblas::bernoulli_sample_seed(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, seed_dist(rand_engine));

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename G, typename T>
struct bernoulli_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G& rand_engine; ///< The custom random engine

public:
    /*!
     * \brief Construct a new bernoulli_unary_g_op
     */
    explicit bernoulli_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_sbernoulli_sample_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dbernoulli_sample_seed);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    auto gpu_compute_hint(const X & x, Y& y) const noexcept {
        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1);
        impl::egblas::bernoulli_sample_seed(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, seed_dist(rand_engine));

        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    Y& gpu_compute(const X & x, Y& y) const noexcept {
        std::uniform_int_distribution<long> seed_dist;

        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1);
        impl::egblas::bernoulli_sample_seed(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, seed_dist(rand_engine));

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename T>
struct state_bernoulli_unary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_bernoulli_sample_prepare)
                                           || (is_double_precision_t<T> && impl::egblas::has_bernoulli_sample_prepare);

private:
    mutable random_engine  rand_engine; ///< The random generator
    std::shared_ptr<void*> states;      ///< The random generator extra states

public:
    /*!
     * \brief Construct a new operator
     */
    state_bernoulli_unary_op() : rand_engine(std::time(nullptr)) {
        if constexpr (impl::egblas::has_bernoulli_sample_prepare) {
            states  = std::make_shared<void*>();
            *states = impl::egblas::bernoulli_sample_prepare();
        }
    }

    /*!
     * \brief Construct a new operator
     */
    explicit state_bernoulli_unary_op(const std::shared_ptr<void*> states) : rand_engine(std::time(nullptr)) {
        if constexpr (impl::egblas::has_bernoulli_sample_prepare) {
            this->states = states;

            if (!*this->states) {
                std::uniform_int_distribution<long> seed_dist;
                *this->states = impl::egblas::bernoulli_sample_prepare_seed(seed_dist(rand_engine));
            }
        }
    }

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    auto gpu_compute_hint(const X & x, Y& y) const noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1);
        impl::egblas::bernoulli_sample_states(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, *states);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    Y& gpu_compute(const X & x, Y& y) const noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1);
        impl::egblas::bernoulli_sample_states(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, *states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename G, typename T>
struct state_bernoulli_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_sbernoulli_sample_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dbernoulli_sample_seed);

private:
    G&                     rand_engine; ///< The custom random engine
    std::shared_ptr<void*> states;      ///< The random generator extra states

public:
    /*!
     * \brief Construct a new state_bernoulli_unary_g_op
     */
    explicit state_bernoulli_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        if constexpr (impl::egblas::has_bernoulli_sample_prepare) {
            std::uniform_int_distribution<long> seed_dist;

            states  = std::make_shared<void*>();
            *states = impl::egblas::bernoulli_sample_prepare_seed(seed_dist(rand_engine));
        }
    }

    /*!
     * \brief Construct a new state_bernoulli_unary_g_op
     */
    state_bernoulli_unary_g_op(G& rand_engine, const std::shared_ptr<void*> & states) : rand_engine(rand_engine) {
        if constexpr (impl::egblas::has_bernoulli_sample_prepare) {
            this->states = states;

            if (!*this->states) {
                std::uniform_int_distribution<long> seed_dist;
                *this->states = impl::egblas::bernoulli_sample_prepare_seed(seed_dist(rand_engine));
            }
        }
    }

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    auto gpu_compute_hint(const X & x, Y& y) const noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1);
        impl::egblas::bernoulli_sample_states(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, *states);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    Y& gpu_compute(const X & x, Y& y) const noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1);
        impl::egblas::bernoulli_sample_states(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, *states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a reverse Bernoulli distribution
 * \tparam T The type of value
 */
template <typename T>
struct reverse_bernoulli_unary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 0.0 : 1.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

/*!
 * \brief Unary operation sampling with a reverse Bernoulli distribution
 * \tparam T The type of value
 */
template <typename G, typename T>
struct reverse_bernoulli_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G& rand_engine; ///< The custom random engine

public:
    /*!
     * \brief Construct a new reverse_bernoulli_unary_g_op
     */
    explicit reverse_bernoulli_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 0.0 : 1.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

} //end of namespace etl
