//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file evaluator.hpp
 * \brief The evaluator is responsible for assigning one expression to another.
 *
 * The evaluator will handle all expressions assignment and for each of them,
 * it will choose the most adapted implementation to assign one to another.
 * There are several implementations of assign:
 *   * standard: Use of standard operators []
 *   * direct: Assign directly to the memory (bypassing the operators)
 *   * fast: memcopy
 *   * vectorized: Use SSE/AVX to compute the expression and store it
 *   * parallel: Parallelized version of direct
 *   * parallel_vectorized  Parallel version of vectorized
 */

/*
 * Possible improvements
 *  * The pre/post functions should be refactored so that is less heavy on the code (too much usage)
 *  * Compound operations should ideally be direct evaluated
 */

#pragma once

#include "etl/eval_selectors.hpp"       //Method selectors
#include "etl/linear_eval_functors.hpp" //Implementation functors
#include "etl/vec_eval_functors.hpp"    //Implementation functors

namespace etl {

/*
 * \brief The evaluator is responsible for assigning one expression to another.
 *
 * The implementation is chosen by SFINAE.
 */
namespace standard_evaluator {
/*!
 * \brief Allocate temporaries and evaluate sub expressions in RHS
 * \param expr The expr to be visited
 */
template <typename E>
void pre_assign_rhs(E&& expr) {
    detail::evaluator_visitor eval_visitor;
    expr.visit(eval_visitor);
}

/*!
 * \brief Assign the result of the expression to the result with the given Functor, using parallel implementation
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Fun, typename E, typename R>
void par_exec(E&& expr, R&& result) {
    if constexpr (parallel_support) {
        auto slice_functor = [&](auto&& lhs, auto&& rhs) { Fun::apply(lhs, rhs); };

        engine_dispatch_1d_slice_binary(result, expr, slice_functor, 0);
    } else {
        Fun::apply(result, expr);
    }
}

// Assign functions implementations

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using the standard [] and read_flat operators.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void standard_assign_impl(E& expr, R& result) {
    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] = expr.read_flat(i);
    }
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using direct memory copy between the two
 * expressions, handling possible GPU memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R, cpp_enable_iff(!is_gpu_dyn_matrix<R>)>
void fast_assign_impl_full(E& expr, R& result) {
    if constexpr (cuda_enabled) {
        cpp_assert(expr.is_cpu_up_to_date() || expr.is_gpu_up_to_date(), "expr must be in valid state");

        if (expr.is_cpu_up_to_date()) {
            direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());

            result.validate_cpu();
        }

        if (expr.is_gpu_up_to_date()) {
            bool cpu_status = expr.is_cpu_up_to_date();

            result.ensure_gpu_allocated();
            result.gpu_copy_from(expr.gpu_memory());

            // Restore CPU status because gpu_copy_from will erase it
            if (cpu_status) {
                result.validate_cpu();
            }
        }

        // Invalidation must be done after validation to preserve
        // valid CPU/GPU state

        if (!expr.is_cpu_up_to_date()) {
            result.invalidate_cpu();
        }

        if (!expr.is_gpu_up_to_date()) {
            result.invalidate_gpu();
        }

        cpp_assert(expr.is_cpu_up_to_date() == result.is_cpu_up_to_date(), "fast_assign must preserve CPU status");
        cpp_assert(expr.is_gpu_up_to_date() == result.is_gpu_up_to_date(), "fast_assign must preserve GPU status");
    } else {
        direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());
    }
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using direct memory copy between the two
 * expressions, handling possible GPU memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R, cpp_enable_iff(is_gpu_dyn_matrix<R>)>
void fast_assign_impl_full([[maybe_unused]] E& expr, [[maybe_unused]] R& result) {
    if constexpr (cuda_enabled) {
        cpp_assert(expr.is_gpu_up_to_date(), "expr must be in valid state");

        result.ensure_gpu_allocated();
        result.gpu_copy_from(expr.gpu_memory());

        // Invalidation must be done after validation to preserve
        // valid CPU/GPU state

        result.validate_gpu();
        result.invalidate_cpu();

        cpp_assert(result.is_gpu_up_to_date(), "fast_assign must preserve GPU status");
    } else {
        cpp_unreachable("gpu_dyn_matrix should never be used without GPU support");
    }
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using direct memory copy between the two
 * expressions.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void fast_assign_impl(E& expr, R& result) {
    static_assert(!is_gpu_dyn_matrix<R>, "gpu_dyn_matrix should not be used here");

    expr.ensure_cpu_up_to_date();

    direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using a full GPU computation.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_assign_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_allocated();

    if constexpr (is_binary_expr<E>) {
        if (expr.alias(result)) {
            // Compute the GPU representation of the expression
            decltype(auto) t2 = smart_gpu_compute_hint(expr, result);

            // Copy the GPU memory from the expression to the result
            result.gpu_copy_from(t2.gpu_memory());
        } else {
            // Compute the GPU representation of the expression into the result
            smart_gpu_compute(expr, result);
        }
    } else {
        // Compute the GPU representation of the expression into the result
        smart_gpu_compute(expr, result);
    }

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using a direct computation and stored in memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void direct_assign_impl(E& expr, R& result) {
    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::Assign>(expr, result);
        } else {
            detail::Assign::apply(result, expr);
        }
    } else {
        detail::Assign::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Assign the result of the expression to the result.
 *
 * This is done using a vectorized computation and stored in
 * memory, possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void vectorized_assign_impl(E& expr, R& result) {
    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    constexpr auto V = detail::select_vector_mode<E, R>();

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::VectorizedAssign<V>>(expr, result);
        } else {
            detail::VectorizedAssign<V>::apply(result, expr);
        }
    } else {
        detail::VectorizedAssign<V>::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

// Selector versions

template <typename E, typename R>
void assign_evaluate_impl_no_gpu(E&& expr, R&& result) {
    if constexpr (detail::standard_assign_no_gpu<E, R>) {
        standard_assign_impl(expr, result);
    } else if constexpr (std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign_no_gpu<E, R>) {
        fast_assign_impl_full(expr, result);
    } else if constexpr (!std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign_no_gpu<E, R>) {
        fast_assign_impl(expr, result);
    } else if constexpr (detail::direct_assign_no_gpu<E, R>) {
        direct_assign_impl(expr, result);
    } else if constexpr (detail::vectorized_assign_no_gpu<E, R>) {
        vectorized_assign_impl(expr, result);
    }
}

template <typename E, typename R>
void assign_evaluate_impl(E&& expr, R&& result) {
    if constexpr (detail::standard_assign<E, R>) {
        standard_assign_impl(expr, result);
    } else if constexpr (std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign<E, R>) {
        fast_assign_impl_full(expr, result);
    } else if constexpr (!std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign<E, R>) {
        fast_assign_impl(expr, result);
    } else if constexpr (detail::gpu_assign<E, R>) {
        if (local_context().cpu || is_something_forced()) {
            assign_evaluate_impl_no_gpu(expr, result);
        } else {
            gpu_assign_impl(expr, result);
        }
    } else if constexpr (detail::direct_assign<E, R>) {
        direct_assign_impl(expr, result);
    } else if constexpr (detail::vectorized_assign<E, R>) {
        vectorized_assign_impl(expr, result);
    }
}

// Compound Assign Add functions implementations

/*!
 * \brief Add the result of the expression to the result.
 *
 * This is performed using standard computation with operator[].
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void standard_compound_add_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] += expr[i];
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Add the result of the expression to the result.
 *
 * This is performed using direct computation with, possibly in
 * parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void direct_compound_add_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::AssignAdd>(expr, result);
        } else {
            detail::AssignAdd::apply(result, expr);
        }
    } else {
        detail::AssignAdd::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Add the result of the expression to the result.
 *
 * This is performed using vectorized computation with, possibly in
 * parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void vectorized_compound_add_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    constexpr auto V = detail::select_vector_mode<E, R>();

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::VectorizedAssignAdd<V>>(expr, result);
        } else {
            detail::VectorizedAssignAdd<V>::apply(result, expr);
        }
    } else {
        detail::VectorizedAssignAdd<V>::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Add the result of the expression to the result.
 *
 * This is performed using full GPU computation with, possibly in
 * parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_add_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    decltype(auto) t1 = smart_gpu_compute_hint(expr, result);

    value_t<E> alpha(1);
    impl::egblas::axpy(etl::size(result), alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

#endif

#ifdef ETL_EGBLAS_MODE

/*!
 * \brief Add the result of the expression to the result.
 *
 * This is performed using full GPU computation with, possibly in
 * parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_add_scalar_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    impl::egblas::scalar_add(result.gpu_memory(), etl::size(result), 1, expr.value);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

#endif

// Selector functions

/*!
 * \brief Add the result of the expression to the result.
 *
 * This does not consider the GPU.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void add_evaluate_no_gpu(E&& expr, R&& result) {
    if constexpr (detail::standard_compound_no_gpu<E, R>) {
        standard_compound_add_impl(expr, result);
    } else if constexpr (detail::direct_compound_no_gpu<E, R>) {
        direct_compound_add_impl(expr, result);
    } else if constexpr (detail::vectorized_compound_no_gpu<E, R>) {
        vectorized_compound_add_impl(expr, result);
    }
}

/*!
 * \brief Add the result of the expression to the result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void add_evaluate(E&& expr, R&& result) {
    if constexpr (detail::standard_compound<E, R>) {
        standard_compound_add_impl(expr, result);
    } else if constexpr (detail::direct_compound<E, R>) {
        direct_compound_add_impl(expr, result);
    } else if constexpr (detail::vectorized_compound<E, R>) {
        vectorized_compound_add_impl(expr, result);
    } else if constexpr (cublas_enabled && detail::gpu_compound<E, R> && !is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            add_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_add_impl(expr, result);
        }
    } else if constexpr (egblas_enabled && detail::gpu_compound<E, R> && is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            add_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_add_scalar_impl(expr, result);
        }
    }
}

// Compound assign sub implementation functions

/*!
 * \brief Subtract the result of the expression from the result
 *
 * This is performed using standard operator[].
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void standard_compound_sub_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] -= expr[i];
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Subtract the result of the expression from the result
 *
 * This is performed using direct compution into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void direct_compound_sub_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::AssignSub>(expr, result);
        } else {
            detail::AssignSub::apply(result, expr);
        }
    } else {
        detail::AssignSub::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Subtract the result of the expression from the result
 *
 * This is performed using vectorized compution into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void vectorized_compound_sub_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    constexpr auto V = detail::select_vector_mode<E, R>();

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::VectorizedAssignSub<V>>(expr, result);
        } else {
            detail::VectorizedAssignSub<V>::apply(result, expr);
        }
    } else {
        detail::VectorizedAssignSub<V>::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Subtract the result of the expression from the result
 *
 * This is performed using full GPU compution into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_sub_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    decltype(auto) t1 = smart_gpu_compute_hint(expr, result);

    value_t<E> alpha(-1);
    impl::egblas::axpy(etl::size(result), alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

#ifdef ETL_EGBLAS_MODE

/*!
 * \brief Subtract the result of the expression from the result
 *
 * This is performed using full GPU compution into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_sub_scalar_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    auto value = -expr.value;
    impl::egblas::scalar_add(result.gpu_memory(), etl::size(result), 1, value);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

#endif

// Selector functions

/*!
 * \brief Subtract the result of the expression from the result.
 *
 * This does not consider the GPU.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void sub_evaluate_no_gpu(E&& expr, R&& result) {
    if constexpr (detail::standard_compound_no_gpu<E, R>) {
        standard_compound_sub_impl(expr, result);
    } else if constexpr (detail::direct_compound_no_gpu<E, R>) {
        direct_compound_sub_impl(expr, result);
    } else if constexpr (detail::vectorized_compound_no_gpu<E, R>) {
        vectorized_compound_sub_impl(expr, result);
    }
}

/*!
 * \brief Subtract the result of the expression from the result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void sub_evaluate(E&& expr, R&& result) {
    if constexpr (detail::standard_compound<E, R>) {
        standard_compound_sub_impl(expr, result);
    } else if constexpr (detail::direct_compound<E, R>) {
        direct_compound_sub_impl(expr, result);
    } else if constexpr (detail::vectorized_compound<E, R>) {
        vectorized_compound_sub_impl(expr, result);
    } else if constexpr (cublas_enabled && detail::gpu_compound<E, R> && !is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            sub_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_sub_impl(expr, result);
        }
    } else if constexpr (egblas_enabled && detail::gpu_compound<E, R> && is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            sub_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_sub_scalar_impl(expr, result);
        }
    }
}

// Compound assign mul implementation functions

/*!
 * \brief Multiply the result by the result of the expression
 *
 * This is performed with standard computation with operator[]
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void standard_compound_mul_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] *= expr[i];
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Multiply the result by the result of the expression
 *
 * This is performed with direct computation into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void direct_compound_mul_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::AssignMul>(expr, result);
        } else {
            detail::AssignMul::apply(result, expr);
        }
    } else {
        detail::AssignMul::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Multiply the result by the result of the expression
 *
 * This is performed with vectorized computation into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void vectorized_compound_mul_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    constexpr auto V = detail::select_vector_mode<E, R>();

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::VectorizedAssignMul<V>>(expr, result);
        } else {
            detail::VectorizedAssignMul<V>::apply(result, expr);
        }
    } else {
        detail::VectorizedAssignMul<V>::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

#ifdef ETL_EGBLAS_MODE

/*!
 * \brief Multiply the result by the result of the expression
 *
 * This is performed with full GPU computation into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_mul_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    decltype(auto) t1 = smart_gpu_compute_hint(expr, result);

    value_t<E> alpha(1);
    impl::egblas::axmy(etl::size(result), alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

#endif

/*!
 * \brief Multiply the result by the result of the expression
 *
 * This is performed with full GPU computation into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_mul_scalar_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    impl::egblas::scalar_mul(result.gpu_memory(), etl::size(result), 1, expr.value);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

// Selector functions

/*!
 * \brief Subtract the result of the expression from the result.
 *
 * This does not consider the GPU.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void mul_evaluate_no_gpu(E&& expr, R&& result) {
    if constexpr (detail::standard_compound_no_gpu<E, R>) {
        standard_compound_mul_impl(expr, result);
    } else if constexpr (detail::direct_compound_no_gpu<E, R>) {
        direct_compound_mul_impl(expr, result);
    } else if constexpr (detail::vectorized_compound_no_gpu<E, R>) {
        vectorized_compound_mul_impl(expr, result);
    }
}

/*!
 * \brief Subtract the result of the expression from the result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void mul_evaluate(E&& expr, R&& result) {
    if constexpr (detail::standard_compound<E, R>) {
        standard_compound_mul_impl(expr, result);
    } else if constexpr (detail::direct_compound<E, R>) {
        direct_compound_mul_impl(expr, result);
    } else if constexpr (detail::vectorized_compound<E, R>) {
        vectorized_compound_mul_impl(expr, result);
    } else if constexpr (egblas_enabled && detail::gpu_compound<E, R> && !is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            mul_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_mul_impl(expr, result);
        }
    } else if constexpr (cublas_enabled && detail::gpu_compound<E, R> && is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            mul_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_mul_scalar_impl(expr, result);
        }
    }
}

// Compound Assign Div implementation functions

/*!
 * \brief Divide the result by the result of the expression
 *
 * This is performed using standard computation with operator[]
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void standard_compound_div_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] /= expr[i];
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Divide the result by the result of the expression
 *
 * This is performed using direct computation into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void direct_compound_div_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::AssignDiv>(expr, result);
        } else {
            detail::AssignDiv::apply(result, expr);
        }
    } else {
        detail::AssignDiv::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Divide the result by the result of the expression
 *
 * This is performed using vectorized computation into memory,
 * possibly in parallel.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void vectorized_compound_div_impl(E& expr, R& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    constexpr auto V = detail::select_vector_mode<E, R>();

    if constexpr (is_thread_safe<E>) {
        if (engine_select_parallel(etl::size(result))) {
            par_exec<detail::VectorizedAssignDiv<V>>(expr, result);
        } else {
            detail::VectorizedAssignDiv<V>::apply(result, expr);
        }
    } else {
        detail::VectorizedAssignDiv<V>::apply(result, expr);
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Divide the result by the result of the expression
 *
 * This is performed using full GPU computation into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_div_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    decltype(auto) t1 = smart_gpu_compute_hint(expr, result);

    value_t<E> alpha(1);
    impl::egblas::axdy(etl::size(result), alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

/*!
 * \brief Divide the result by the result of the expression
 *
 * This is performed using full GPU computation into memory.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void gpu_compound_div_scalar_impl(E& expr, R& result) {
    inc_counter("gpu:assign");

    result.ensure_gpu_up_to_date();

    // Compute the GPU representation of the expression
    auto value = value_t<E>(1.0) / expr.value;
    impl::egblas::scalar_mul(result.gpu_memory(), etl::size(result), 1, value);

    // Validate the GPU and invalidates the CPU
    result.validate_gpu();
    result.invalidate_cpu();
}

// Selector functions

/*!
 * \brief Divide the result by the result of the expression.
 *
 * This does not consider the GPU.
 *
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void div_evaluate_no_gpu(E&& expr, R&& result) {
    if constexpr (detail::standard_compound_div_no_gpu<E, R>) {
        standard_compound_div_impl(expr, result);
    } else if constexpr (detail::direct_compound_div_no_gpu<E, R>) {
        direct_compound_div_impl(expr, result);
    } else if constexpr (detail::vectorized_compound_div_no_gpu<E, R>) {
        vectorized_compound_div_impl(expr, result);
    }
}

/*!
 * \brief Divide the result by the result of the expression
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void div_evaluate(E&& expr, R&& result) {
    if constexpr (detail::standard_compound_div<E, R>) {
        standard_compound_div_impl(expr, result);
    } else if constexpr (detail::direct_compound_div<E, R>) {
        direct_compound_div_impl(expr, result);
    } else if constexpr (detail::vectorized_compound_div<E, R>) {
        vectorized_compound_div_impl(expr, result);
    } else if constexpr (egblas_enabled && detail::gpu_compound_div<E, R> && !is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            div_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_div_impl(expr, result);
        }
    } else if constexpr (cublas_enabled && detail::gpu_compound_div<E, R> && is_scalar<E>) {
        if (local_context().cpu || is_something_forced()) {
            div_evaluate_no_gpu(expr, result);
        } else {
            gpu_compound_div_scalar_impl(expr, result);
        }
    }
}

//Standard Mod Evaluate (no optimized versions for mod)

/*!
 * \brief Modulo the result by the result of the expression
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void mod_evaluate(E&& expr, R&& result) {
    pre_assign_rhs(expr);

    safe_ensure_cpu_up_to_date(expr);
    safe_ensure_cpu_up_to_date(result);

    for (size_t i = 0; i < etl::size(result); ++i) {
        result[i] %= expr[i];
    }

    result.validate_cpu();
    result.invalidate_gpu();
}

/*!
 * \brief Assign the result of the expression to the result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename E, typename R>
void assign_evaluate(E&& expr, R&& result) {
    if constexpr (!detail::gpu_assign<E, R>) {
        //Evaluate sub parts, if any
        pre_assign_rhs(expr);
    }

    //Perform the real evaluation, selected by TMP
    assign_evaluate_impl(expr, result);
}

} // end of namespace standard_evaluator

/*!
 * \brief Traits indicating if a direct assign is possible
 *
 * A direct assign is a standard assign without any transposition
 *
 * \tparam Expr The type of expression (RHS)
 * \tparam Result The type of result (LHS)
 */
template <typename Expr, typename Result>
constexpr bool direct_assign_compatible = decay_traits<Expr>::is_generator // No dimensions, always possible to assign
                                          || decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order // Same storage always possible to assign
                                          || all_1d<Expr, Result> // Vectors can be directly assigned, regardless of the storage order
    ;

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_assign_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::assign_evaluate(expr, result);
    } else {
        standard_evaluator::assign_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_add_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::add_evaluate(expr, result);
    } else {
        standard_evaluator::add_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_sub_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::sub_evaluate(expr, result);
    } else {
        standard_evaluator::sub_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_mul_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::mul_evaluate(expr, result);
    } else {
        standard_evaluator::mul_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_div_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::div_evaluate(expr, result);
    } else {
        standard_evaluator::div_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result>
void std_mod_evaluate(Expr&& expr, Result&& result) {
    if constexpr (direct_assign_compatible<Expr, Result>) {
        standard_evaluator::mod_evaluate(expr, result);
    } else {
        standard_evaluator::mod_evaluate(transpose(expr), result);
    }
}

/*!
 * \brief Force the internal evaluation of an expression
 * \param expr The expression to force inner evaluation
 *
 * This function can be used when complex expressions are used
 * lazily.
 */
template <typename Expr>
void force(Expr&& expr) {
    standard_evaluator::pre_assign_rhs(expr);
}

} //end of namespace etl
