//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

#pragma once

#include "etl/parallel.hpp"       //Parallel helpers
#include "etl/traits_lite.hpp"    //forward declaration of the traits
#include "etl/visitor.hpp"        //visitor of the expressions
#include "etl/threshold.hpp"      //parallel thresholds
#include "etl/eval_selectors.hpp" //method selectors
#include "etl/eval_functors.hpp"  //Implementation functors
#include "etl/eval_visitors.hpp"  //Evaluation visitors

namespace etl {

template <typename Expr, typename Result>
struct standard_evaluator {
    template <typename E>
    static void evaluate_only(E&& expr) {
        apply_visitor<detail::temporary_allocator_static_visitor>(expr);
        apply_visitor<detail::evaluator_static_visitor>(expr);
    }

    //Standard assign version

    template <typename E, typename R, cpp_enable_if(detail::standard_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] = expr.read_flat(i);
        }
    }

    //Fast assign version (memory copy)

    template <typename E, typename R, cpp_enable_if(detail::fast_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        std::copy(expr.memory_start(), expr.memory_end(), result.memory_start());
    }

    //Direct assign version

    template <typename E, typename R>
    static void direct_assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::Assign<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(detail::direct_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        direct_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel assign version

    template <typename E, typename R, cpp_enable_if(detail::parallel_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2){
            direct_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        evaluate_only(expr);

        auto m = result.memory_start();

        static cpp::default_thread_pool<> pool(threads - 1);

        //Distribute evenly the batches

        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::Assign<value_t<R>,E>(m, expr, t * batch, (t+1) * batch));
        }

        detail::Assign<value_t<R>,E>(m, expr, (threads - 1) * batch, n)();

        pool.wait();
    }

    //Parallel vectorized assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssign<R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssign<R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized assign version

    template <typename E, typename R>
    static void vectorized_assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        detail::VectorizedAssign<R, E>(result, expr, 0, etl::size(result))();
    }

    template <typename E, typename R, cpp_enable_if(detail::vectorized_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Standard Add Assign

    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] += expr[i];
        }
    }

    //Parallel direct add assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2){
            direct_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        evaluate_only(expr);

        auto m = result.memory_start();

        static cpp::default_thread_pool<> pool(threads - 1);

        //Distribute evenly the batches

        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::AssignAdd<value_t<R>,E>(m, expr, t * batch, (t+1) * batch));
        }

        detail::AssignAdd<value_t<R>,E>(m, expr, (threads - 1) * batch, n)();

        pool.wait();
    }

    //Direct Add Assign

    template <typename E, typename R>
    static void direct_add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignAdd<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        direct_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized add assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignAdd<R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignAdd<R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized Add Assign

    template <typename E, typename R>
    static void vectorized_add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        detail::VectorizedAssignAdd<R, E>(result, expr, 0, etl::size(result))();
    }

    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        vectorized_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Standard sub assign

    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] -= expr[i];
        }
    }

    //Parallel direct sub assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2){
            direct_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        evaluate_only(expr);

        auto m = result.memory_start();

        static cpp::default_thread_pool<> pool(threads - 1);

        //Distribute evenly the batches

        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::AssignSub<value_t<R>,E>(m, expr, t * batch, (t+1) * batch));
        }

        detail::AssignSub<value_t<R>,E>(m, expr, (threads - 1) * batch, n)();

        pool.wait();
    }

    //Direct Sub Assign

    template <typename E, typename R>
    static void direct_sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignSub<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        direct_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized sub assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignSub<R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignSub<R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized Sub Assign

    template <typename E, typename R>
    static void vectorized_sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        detail::VectorizedAssignSub<R, E>(result, expr, 0, etl::size(result))();
    }

    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        vectorized_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Standard Mul Assign

    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] *= expr[i];
        }
    }

    //Parallel direct mul assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2){
            direct_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        evaluate_only(expr);

        auto m = result.memory_start();

        static cpp::default_thread_pool<> pool(threads - 1);

        //Distribute evenly the batches

        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::AssignMul<value_t<R>,E>(m, expr, t * batch, (t+1) * batch));
        }

        detail::AssignMul<value_t<R>,E>(m, expr, (threads - 1) * batch, n)();

        pool.wait();
    }

    //Direct Mul Assign

    template <typename E, typename R>
    static void direct_mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignMul<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        direct_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized mul assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignMul<R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignMul<R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized Mul Assign

    template <typename E, typename R>
    static void vectorized_mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        detail::VectorizedAssignMul<R, E>(result, expr, 0, etl::size(result))();
    }

    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        vectorized_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Standard Div Assign

    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] /= expr[i];
        }
    }

    //Parallel direct Div assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2){
            direct_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        evaluate_only(expr);

        auto m = result.memory_start();

        static cpp::default_thread_pool<> pool(threads - 1);

        //Distribute evenly the batches

        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::AssignDiv<value_t<R>,E>(m, expr, t * batch, (t+1) * batch));
        }

        detail::AssignDiv<value_t<R>,E>(m, expr, (threads - 1) * batch, n)();

        pool.wait();
    }

    //Direct Div Assign

    template <typename E, typename R>
    static void direct_div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignDiv<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        direct_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized div assign

    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignDiv<R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignDiv<R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized Div Assign

    template <typename E, typename R>
    static void vectorized_div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        detail::VectorizedAssignDiv<R, E>(result, expr, 0, etl::size(result))();
    }

    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        vectorized_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Standard Mod Evaluate (no optimized versions for mod)

    template <typename E, typename R>
    static void mod_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] %= expr[i];
        }
    }

    //Note: In case of direct evaluation, the temporary_expr itself must
    //not beevaluated by the static_visitor, otherwise, the result would
    //be evaluated twice and a temporary would be allocated for nothing

    template <typename E, typename R, cpp_enable_if(is_temporary_unary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.a());
        apply_visitor<detail::evaluator_static_visitor>(expr.a());

        expr.direct_evaluate(result);
    }

    template <typename E, typename R, cpp_enable_if(is_temporary_binary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.a());
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.b());

        apply_visitor<detail::evaluator_static_visitor>(expr.a());
        apply_visitor<detail::evaluator_static_visitor>(expr.b());

        expr.direct_evaluate(result);
    }
};

//Only containers of the same storage order can be assigned directly
//Generators can be assigned to everything
template <typename Expr, typename Result>
struct direct_assign_compatible : cpp::or_u<
                                      decay_traits<Expr>::is_generator,
                                      decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order> {};

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value && !is_optimized_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::assign_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value && !is_optimized_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::assign_evaluate(transpose(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(), [&result](const auto& optimized) { assign_evaluate(optimized, std::forward<Result>(result)); });
}

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::add_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::add_evaluate(transpose(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::sub_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::sub_evaluate(transpose(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::mul_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::mul_evaluate(transpose(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::div_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::div_evaluate(transpose(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::mod_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template <typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator<Expr, Result>::mod_evaluate(transpose(expr), std::forward<Result>(result));
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
    standard_evaluator<Expr, void>::evaluate_only(std::forward<Expr>(expr));
}

} //end of namespace etl
