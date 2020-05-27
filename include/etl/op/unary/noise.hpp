//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/logistic_noise.hpp"

namespace etl {

/*!
 * \brief Unary operation applying an uniform noise (0.0, 1.0(
 * \tparam T The type of value
 */
template <typename T>
struct uniform_noise_unary_op {
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
        static std::uniform_real_distribution<double> real_distribution(0.0, 1.0);

        return x + real_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

/*!
 * \brief Unary operation applying an uniform noise (0.0, 1.0(
 * \tparam T The type of value
 */
template <typename G, typename T>
struct uniform_noise_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G& rand_engine; ///< The custom random engine

public:
    /*!
     * \brief Construct a new uniform_noise_unary_g_op
     */
    explicit uniform_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
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
        std::uniform_real_distribution<double> real_distribution(0.0, 1.0);

        return x + real_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

/*!
 * \brief Unary operation applying a normal noise
 * \tparam T The type of value
 */
template <typename T>
struct normal_noise_unary_op {
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
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);

        return x + normal_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "normal_noise";
    }
};

/*!
 * \brief Unary operation applying a normal noise
 * \tparam T The type of value
 */
template <typename G, typename T>
struct normal_noise_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G& rand_engine; ///< The custom random engine

public:
    /*!
     * \brief Construct a new normal_noise_unary_g_op
     */
    explicit normal_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
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
        std::normal_distribution<double> normal_distribution(0.0, 1.0);

        return x + normal_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "normal_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise
 * \tparam T The type of value
 */
template <typename T>
struct logistic_noise_unary_op {
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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_slogistic_noise_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dlogistic_noise_seed);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));

        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
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
        impl::egblas::logistic_noise_seed(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, seed_dist(rand_engine));

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
        impl::egblas::logistic_noise_seed(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, seed_dist(rand_engine));

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise
 * \tparam T The type of value
 */
template <typename G, typename T>
struct logistic_noise_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G& rand_engine; ///< The custom random engine

public:
    /*!
     * \brief Construct a new logistic_noise_unary_g_op
     */
    explicit logistic_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_slogistic_noise_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dlogistic_noise_seed);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
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
        impl::egblas::logistic_noise_seed(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, seed_dist(rand_engine));

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
        impl::egblas::logistic_noise_seed(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, seed_dist(rand_engine));

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise
 *
 * This version is stateful in that the random generator states are created
 * only once. This has an improvement on GPU.
 *
 * \tparam T The type of value
 */
template <typename T>
struct state_logistic_noise_unary_op {
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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_slogistic_noise_seed)
                                           || (is_double_precision_t<T> && impl::egblas::has_dlogistic_noise_seed);

    mutable random_engine  rand_engine; ///< The random generator
    std::shared_ptr<void*> states;      ///< The random generator extra states

    /*!
     * \brief Construct a new operator
     */
    state_logistic_noise_unary_op() : rand_engine(std::time(nullptr)) {
        if constexpr (impl::egblas::has_logistic_noise_prepare) {
            states  = std::make_shared<void*>();
            *states = impl::egblas::logistic_noise_prepare();
        }
    }

    /*!
     * \brief Construct a new operator
     */
    explicit state_logistic_noise_unary_op(const std::shared_ptr<void*> states) : rand_engine(std::time(nullptr)) {
        if constexpr (impl::egblas::has_logistic_noise_prepare) {
            this->states = states;

            if (!*this->states) {
                std::uniform_int_distribution<long> seed_dist;
                *this->states = impl::egblas::logistic_noise_prepare_seed(seed_dist(rand_engine));
            }
        }
    }

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
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
        impl::egblas::logistic_noise_states(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, *states);

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
        impl::egblas::logistic_noise_states(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, *states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "state_logistic_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise. 
 *
 * This version is stateful in that the random generator states are created
 * only once. This has an improvement on GPU.
 *
 * \tparam T The type of value
 */
template <typename G, typename T>
struct state_logistic_noise_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:
    G&                     rand_engine; ///< The custom random engine
    std::shared_ptr<void*> states;      ///< The random generator extra states

public:
    /*!
     * \brief Construct a new state_logistic_noise_unary_g_op
     */
    explicit state_logistic_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        if constexpr (impl::egblas::has_logistic_noise_prepare) {
            std::uniform_int_distribution<long> seed_dist;

            states  = std::make_shared<void*>();
            *states = impl::egblas::logistic_noise_prepare_seed(seed_dist(rand_engine));
        }
    }

    /*!
     * \brief Construct a new state_logistic_noise_unary_g_op
     */
    state_logistic_noise_unary_g_op(G& rand_engine, const std::shared_ptr<void*> & states) : rand_engine(rand_engine) {
        if constexpr (impl::egblas::has_logistic_noise_prepare) {
            this->states = states;

            if (!*this->states) {
                std::uniform_int_distribution<long> seed_dist;
                *this->states = impl::egblas::logistic_noise_prepare_seed(seed_dist(rand_engine));
            }
        }
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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_slogistic_noise_states)
                                           || (is_double_precision_t<T> && impl::egblas::has_dlogistic_noise_states);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
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
        impl::egblas::logistic_noise_states(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1, *states);

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
        impl::egblas::logistic_noise_states(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1, *states);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "state_logistic_noise";
    }
};

} //end of namespace etl
