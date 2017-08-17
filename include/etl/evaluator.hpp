//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

#include "cpp_utils/static_if.hpp"

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

    //Standard assign version

    /*!
     * \brief Assign the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_iff(detail::standard_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] = expr.read_flat(i);
        }
    }

    //Fast assign version (memory copy)

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_iff(std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
//TODO(CPP17) if constexpr
#ifdef ETL_CUDA
        if(expr.is_cpu_up_to_date()){
            direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());

            result.validate_cpu();
        } else {
            result.invalidate_cpu();
        }

        if(expr.is_gpu_up_to_date()){
            bool cpu_status = result.is_cpu_up_to_date();

            result.ensure_gpu_allocated();
            result.gpu_copy_from(expr.gpu_memory());

            result.validate_gpu();

            // Restore CPU status because gpu_copy_from will erase it
            if(cpu_status){
                result.validate_cpu();
            }
        } else {
            result.invalidate_gpu();
        }
#else
        direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());
#endif
    }

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_iff(!std::is_same<value_t<E>, value_t<R>>::value && detail::fast_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        expr.ensure_cpu_up_to_date();

        direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());

        result.invalidate_gpu();
    }

    // GPU assign version

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_iff(detail::gpu_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        inc_counter("gpu:assign");

        result.ensure_gpu_allocated();

        // Compute the GPU representation of the expression
        auto t1 = expr.gpu_compute();

        // Copy the GPU memory from the expression to the result
        result.gpu_copy_from(t1.gpu_memory());

        // Validate the GPU and invalidates the CPU
        result.validate_gpu();
        result.invalidate_cpu();
    }

    //Parallel assign version

// CPP17: if constexpr here
#ifdef ETL_PARALLEL_SUPPORT
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename Fun, typename E, typename R>
    void par_exec(E&& expr, R&& result) {
        auto slice_functor = [&](auto&& lhs, auto&& rhs){
            Fun::apply(lhs, rhs);
        };

        engine_dispatch_1d_slice_binary(result, expr, slice_functor, 0);
    }
#else
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename Fun, typename E, typename R>
    void par_exec(E&& expr, R&& result) {
        Fun::apply(result, expr);
    }
#endif

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_iff(detail::direct_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::Assign>(expr, result);
        } else {
            detail::Assign::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_iff(detail::vectorized_assign<E, R>)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        constexpr auto V = detail::select_vector_mode<E, R>();

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::VectorizedAssign<V>>(expr, result);
        } else {
            detail::VectorizedAssign<V>::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    //Standard Add Assign

    /*!
     * \brief Add the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_iff(detail::standard_compound<E, R>)>
    void add_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] += expr[i];
        }

        result.invalidate_gpu();
    }

    //Parallel direct add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::direct_compound<E, R>)>
    void add_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::AssignAdd>(expr, result);
        } else {
            detail::AssignAdd::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::vectorized_compound<E, R>)>
    void add_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::VectorizedAssignAdd<V>>(expr, result);
        } else {
            detail::VectorizedAssignAdd<V>::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    // GPU assign add version

#ifdef ETL_CUBLAS_MODE

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::gpu_compound<E, R>)>
    void add_evaluate(E&& expr, R&& result) {
        inc_counter("gpu:assign");

        pre_assign_rhs(expr);

        result.ensure_gpu_up_to_date();

        // Compute the GPU representation of the expression
        auto t1 = expr.gpu_compute();

        decltype(auto) handle = impl::cublas::start_cublas();

        value_t<E> alpha(1);
        impl::cublas::cublas_axpy(handle.get(), size(result), &alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

        // Validate the GPU and invalidates the CPU
        result.validate_gpu();
        result.invalidate_cpu();
    }

#endif

    //Standard sub assign

    /*!
     * \brief Subtract the result of the expression expression from the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_iff(detail::standard_compound<E, R>)>
    void sub_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] -= expr[i];
        }

        result.invalidate_gpu();
    }

    //Parallel direct sub assign

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::direct_compound<E, R>)>
    void sub_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::AssignSub>(expr, result);
        } else {
            detail::AssignSub::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized sub assign

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::vectorized_compound<E, R>)>
    void sub_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::VectorizedAssignSub<V>>(expr, result);
        } else {
            detail::VectorizedAssignSub<V>::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    // GPU assign add version

#ifdef ETL_CUBLAS_MODE

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::gpu_compound<E, R>)>
    void sub_evaluate(E&& expr, R&& result) {
        inc_counter("gpu:assign");

        pre_assign_rhs(expr);

        result.ensure_gpu_up_to_date();

        // Compute the GPU representation of the expression
        auto t1 = expr.gpu_compute();

        decltype(auto) handle = impl::cublas::start_cublas();

        value_t<E> alpha(-1);
        impl::cublas::cublas_axpy(handle.get(), size(result), &alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

        // Validate the GPU and invalidates the CPU
        result.validate_gpu();
        result.invalidate_cpu();
    }

#endif

    //Standard Mul Assign

    /*!
     * \brief Multiply the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_iff(detail::standard_compound<E, R>)>
    void mul_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] *= expr[i];
        }

        result.invalidate_gpu();
    }

    //Parallel direct mul assign

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::direct_compound<E, R>)>
    void mul_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::AssignMul>(expr, result);
        } else {
            detail::AssignMul::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized mul assign

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::vectorized_compound<E, R>)>
    void mul_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::VectorizedAssignMul<V>>(expr, result);
        } else {
            detail::VectorizedAssignMul<V>::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    // GPU assign mul version

#ifdef ETL_EGBLAS_MODE

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::gpu_compound<E, R>)>
    void mul_evaluate(E&& expr, R&& result) {
        inc_counter("gpu:assign");

        pre_assign_rhs(expr);

        result.ensure_gpu_up_to_date();

        // Compute the GPU representation of the expression
        auto t1 = expr.gpu_compute();

        value_t<E> alpha(1);
        impl::egblas::axmy(size(result), &alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

        // Validate the GPU and invalidates the CPU
        result.validate_gpu();
        result.invalidate_cpu();
    }

#endif

    //Standard Div Assign

    /*!
     * \brief Divide the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_iff(detail::standard_compound_div<E, R>)>
    void div_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] /= expr[i];
        }

        result.invalidate_gpu();
    }

    //Parallel direct Div assign

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::direct_compound_div<E, R>)>
    void div_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::AssignDiv>(expr, result);
        } else {
            detail::AssignDiv::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized div assign

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::vectorized_compound_div<E, R>)>
    void div_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(is_thread_safe<E> && select_parallel(etl::size(result))){
            par_exec<detail::VectorizedAssignDiv<V>>(expr, result);
        } else {
            detail::VectorizedAssignDiv<V>::apply(result, expr);
        }

        result.invalidate_gpu();
    }

    // GPU assign div version

#ifdef ETL_EGBLAS_MODE

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_iff(detail::gpu_compound_div<E, R>)>
    void div_evaluate(E&& expr, R&& result) {
        inc_counter("gpu:assign");

        pre_assign_rhs(expr);

        result.ensure_gpu_up_to_date();

        // Compute the GPU representation of the expression
        auto t1 = expr.gpu_compute();

        value_t<E> alpha(1);
        impl::egblas::axdy(size(result), &alpha, t1.gpu_memory(), 1, result.gpu_memory(), 1);

        // Validate the GPU and invalidates the CPU
        result.validate_gpu();
        result.invalidate_cpu();
    }

#endif

    //Standard Mod Evaluate (no optimized versions for mod)

    /*!
     * \brief Modulo the result by the result of the expression expression
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

        result.invalidate_gpu();
    }

    /*!
     * \brief Assign the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R>
    void assign_evaluate(E&& expr, R&& result) {
        //Evaluate sub parts, if any
        pre_assign_rhs(expr);

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
constexpr bool direct_assign_compatible =
    decay_traits<Expr>::is_generator || decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order;

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(expr, result);
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(expr, result);
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(expr, result);
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(expr, result);
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(expr, result);
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(direct_assign_compatible<Expr, Result>)>
void std_mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mod_evaluate(expr, result);
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_iff(!direct_assign_compatible<Expr, Result>)>
void std_mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mod_evaluate(transpose(expr), result);
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
