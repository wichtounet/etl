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
 *   * direct: Assign directly to the memory
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

#include "etl/eval_selectors.hpp" //method selectors
#include "etl/eval_functors.hpp"  //Implementation functors

// Optimized evaluations
#include "etl/impl/transpose.hpp"

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
        expr.visit(detail::temporary_allocator_visitor{});

        detail::evaluator_visitor eval_visitor;
        expr.visit(eval_visitor);

        expr.visit(detail::back_propagate_visitor{});
    }

    /*!
     * \brief Allocate temporaries and evaluate sub expressions in LHS
     * \param expr The expr to be visited
     */
    template <typename E>
    void pre_assign_lhs(E&& expr) {
        expr.visit(detail::back_propagate_visitor{});
    }

    //Standard assign version

    /*!
     * \brief Assign the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] = expr.read_flat(i);
        }
    }

    //Fast assign version (memory copy)

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(std::is_same<value_t<E>, value_t<R>>::value, detail::fast_assign<E, R>::value)>
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
    template <typename E, typename R, cpp_enable_if(!std::is_same<value_t<E>, value_t<R>>::value, detail::fast_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        expr.ensure_cpu_up_to_date();

        direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());

        result.invalidate_gpu();
    }

    //Parallel assign version

// CPP17: if constexpr here
#ifdef ETL_PARALLEL_SUPPORT
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel non-vectorized implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <template <typename, typename> class Fun, typename E, typename R>
    void par_linear(E&& expr, R&& result) {
        const auto n = etl::size(result);

        using RS = decltype(memory_slice(result, 0, n));
        using ES = decltype(memory_slice(expr, 0, n));

        ETL_PARALLEL_SESSION {
            thread_engine::acquire();

            //Distribute evenly the batches

            auto batch = n / threads;

            for (size_t t = 0; t < threads - 1; ++t) {
                thread_engine::schedule(Fun<RS, ES>(memory_slice(result, t * batch, (t + 1) * batch), memory_slice(expr, t * batch, (t + 1) * batch)));
            }

            thread_engine::schedule(Fun<RS, ES>(memory_slice(result, (threads - 1) * batch, n), memory_slice(expr, (threads - 1) * batch, n)));

            thread_engine::wait();
        }
    }
#else
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel non-vectorized implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <template <typename, typename> class Fun, typename E, typename R>
    void par_linear(E&& expr, R&& result) {
        Fun<R, E>(result, expr)();
    }
#endif

// CPP17: if constexpr here
#ifdef ETL_PARALLEL_SUPPORT
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel vectorized implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <template <vector_mode_t, typename, typename> class Fun, vector_mode_t V, typename E, typename R>
    void par_vec(E&& expr, R&& result) {
        const auto n = etl::size(result);

        using RS = decltype(memory_slice(result, 0, n));
        using ES = decltype(memory_slice(expr, 0, n));

        ETL_PARALLEL_SESSION {
            thread_engine::acquire();

            //Distribute evenly the batches

            auto batch = n / threads;

            for (size_t t = 0; t < threads - 1; ++t) {
                thread_engine::schedule(Fun<V, RS, ES>(memory_slice(result, t * batch, (t + 1) * batch), memory_slice(expr, t * batch, (t + 1) * batch)));
            }

            thread_engine::schedule(Fun<V, RS, ES>(memory_slice(result, (threads - 1) * batch, n), memory_slice(expr, (threads - 1) * batch, n)));

            thread_engine::wait();
        }
    }
#else
    /*!
     * \brief Assign the result of the expression expression to the result with the given Functor, using parallel vectorized implementation
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <template <vector_mode_t, typename, typename> class Fun, vector_mode_t V, typename E, typename R>
    void par_vec(E&& expr, R&& result) {
        Fun<V, R&, E&>(result, expr)();
    }
#endif

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_linear<detail::Assign>(expr, result);
        } else {
            detail::Assign<R&,E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        constexpr auto V = detail::select_vector_mode<E, R>();

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_vec<detail::VectorizedAssign, V>(expr, result);
        } else {
            detail::VectorizedAssign<V, R&, E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Standard Add Assign

    /*!
     * \brief Add the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        for (size_t i = 0; i < etl::size(result); ++i) {
            result[i] += expr[i];
        }

        result.invalidate_gpu();
    }

    //Parallel direct add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_linear<detail::AssignAdd>(expr, result);
        } else {
            detail::AssignAdd<R&,E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_vec<detail::VectorizedAssignAdd, V>(expr, result);
        } else {
            detail::VectorizedAssignAdd<V, R&, E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Standard sub assign

    /*!
     * \brief Subtract the result of the expression expression from the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

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
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_linear<detail::AssignSub>(expr, result);
        } else {
            detail::AssignSub<R&,E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized sub assign

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_vec<detail::VectorizedAssignSub, V>(expr, result);
        } else {
            detail::VectorizedAssignSub<V, R&, E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Standard Mul Assign

    /*!
     * \brief Multiply the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

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
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_linear<detail::AssignMul>(expr, result);
        } else {
            detail::AssignMul<R&,E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized mul assign

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_vec<detail::VectorizedAssignMul, V>(expr, result);
        } else {
            detail::VectorizedAssignMul<V, R&, E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Standard Div Assign

    /*!
     * \brief Divide the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound_div<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

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
    template <typename E, typename R, cpp_enable_if(detail::direct_compound_div<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_linear<detail::AssignDiv>(expr, result);
        } else {
            detail::AssignDiv<R&,E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Parallel vectorized div assign

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound_div<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        constexpr auto V = detail::select_vector_mode<E, R>();

        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        safe_ensure_cpu_up_to_date(expr);
        safe_ensure_cpu_up_to_date(result);

        if(all_thread_safe<E>::value && select_parallel(etl::size(result))){
            par_vec<detail::VectorizedAssignDiv, V>(expr, result);
        } else {
            detail::VectorizedAssignDiv<V, R&, E&>(result, expr)();
        }

        result.invalidate_gpu();
    }

    //Standard Mod Evaluate (no optimized versions for mod)

    /*!
     * \brief Modulo the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R>
    void mod_evaluate(E&& expr, R&& result) {
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

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
    template <typename E, typename R, cpp_enable_if(decay_traits<E>::is_linear)>
    void assign_evaluate(E&& expr, R&& result) {
        //Evaluate sub parts, if any
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        //Perform the real evaluation, selected by TMP
        assign_evaluate_impl(expr, result);
    }

    /*!
     * \copydoc assign_evaluate
     */
    template <typename E, typename R, cpp_enable_if(!decay_traits<E>::is_linear)>
    void assign_evaluate(E&& expr, R&& result) {
        //Evaluate sub parts, if any
        pre_assign_rhs(expr);
        pre_assign_lhs(result);

        if(result.alias(expr)){
            auto tmp_result = force_temporary(result);

            //Perform the evaluation to tmp_result
            assign_evaluate_impl(expr, tmp_result);

            //Perform the real evaluation to result
            assign_evaluate_impl(tmp_result, result);
        } else {
            //Perform the real evaluation, selected by TMP
            assign_evaluate_impl(expr, result);
        }
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
struct direct_assign_compatible : cpp::or_u<
                                      decay_traits<Expr>::is_generator,
                                      decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order> {};

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(expr, result);
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
void std_assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(expr, result);
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
void std_add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(expr, result);
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
void std_sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(expr, result);
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
void std_mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(expr, result);
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
void std_div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(transpose(expr), result);
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void std_mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mod_evaluate(expr, result);
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value)>
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
