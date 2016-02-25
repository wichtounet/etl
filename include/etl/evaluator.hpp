//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

#include "etl/visitor.hpp"        //visitor of the expressions
#include "etl/threshold.hpp"      //parallel thresholds
#include "etl/eval_selectors.hpp" //method selectors
#include "etl/eval_functors.hpp"  //Implementation functors
#include "etl/eval_visitors.hpp"  //Evaluation visitors

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
     * \brief Performs a direct memory copy
     * \param first pointer to the first element to copy
     * \param last pointer to the next-to-last element to copy
     * \param target pointer to the first element of the result
     */
    template <typename S, typename T>
    void direct_copy(const S* first, const S* last, T* target) {
        std::copy(first, last, target);
    }

    /*!
     * \brief Allocate temporaries and evaluate sub expressions
     * \param expr The expr to be visited
     */
    template <typename E>
    void pre_assign(E&& expr) {
        apply_visitor<detail::temporary_allocator_static_visitor>(expr);
        apply_visitor<detail::evaluator_static_visitor>(expr);
    }

    template <typename E, typename R>
    void post_assign(E&& expr, R&& result) {
        //TODO This is probably a bit overcomplicated
#ifdef ETL_CUDA
        //If necessary copy the GPU result back to CPU
        cpp::static_if<all_dma<R>::value && !etl::is_sparse_matrix<R>::value>([&](auto f){
            f(result).gpu_copy_from_if_necessary();
        });
#endif

        apply_visitor<detail::gpu_clean_static_visitor>(expr);
        apply_visitor<detail::gpu_clean_static_visitor>(result);
    }

    template <typename E>
    void post_assign_compound(E&& expr) {
        //TODO This is probably a bit overcomplicated
#ifdef ETL_CUDA
        //If necessary copy the GPU result back to CPU
        cpp::static_if<all_dma<E>::value && !etl::is_sparse_matrix<E>::value>([&](auto f){
            f(expr).gpu_copy_from_if_necessary();
        });
#endif

        apply_visitor<detail::gpu_clean_static_visitor>(expr);
    }

    template <typename E>
    void post_assign_force(E&& expr) {
        post_assign_compound(expr);
    }

    //Standard assign version

    /*!
     * \brief Assign the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] = expr.read_flat(i);
        }
    }

    //Fast assign version (memory copy)

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::fast_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        direct_copy(expr.memory_start(), expr.memory_end(), result.memory_start());
    }

    //Direct assign version

    template <typename E, typename R>
    void direct_assign_evaluate(E&& expr, R&& result) {
        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::Assign<value_t<R>,E>(m, expr, 0, size)();
    }

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        direct_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel assign version

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2 || local_context().serial){
            direct_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

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

    //Vectorized assign version

    template <typename E, typename R>
    void vectorized_assign_evaluate(E&& expr, R&& result) {
        detail::VectorizedAssign<detail::select_vector_mode<E, R>(), R, E>(result, expr, 0, etl::size(result))();
    }

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized assign

    /*!
     * \copydoc assign_evaluate_impl
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_assign<E, R>::value)>
    void assign_evaluate_impl(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2 || local_context().serial){
            vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssign<detail::select_vector_mode<E, R>(), R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssign<detail::select_vector_mode<E, R>(), R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Standard Add Assign

    /*!
     * \brief Add the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] += expr[i];
        }
    }

    //Direct Add Assign

    template <typename E, typename R>
    void direct_add_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignAdd<value_t<R>,E>(m, expr, 0, size)();
    }

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        direct_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel direct add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2 || local_context().serial){
            direct_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        pre_assign(expr);
        post_assign_compound(expr);

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

    //Vectorized Add Assign

    template <typename E, typename R>
    void vectorized_add_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        detail::VectorizedAssignAdd<detail::select_vector_mode<E, R>(), R, E>(result, expr, 0, etl::size(result))();
    }

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        vectorized_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized add assign

    /*!
     * \copydoc add_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    void add_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2 || local_context().serial){
            vectorized_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        pre_assign(expr);
        post_assign_compound(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignAdd<detail::select_vector_mode<E, R>(), R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignAdd<detail::select_vector_mode<E, R>(), R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Standard sub assign

    /*!
     * \brief Subtract the result of the expression expression from the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] -= expr[i];
        }
    }

    //Direct Sub Assign

    template <typename E, typename R>
    void direct_sub_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignSub<value_t<R>,E>(m, expr, 0, size)();
    }

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        direct_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel direct sub assign

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2 || local_context().serial){
            direct_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        pre_assign(expr);
        post_assign_compound(expr);

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

    //Vectorized Sub Assign

    template <typename E, typename R>
    void vectorized_sub_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        detail::VectorizedAssignSub<detail::select_vector_mode<E, R>(), R, E>(result, expr, 0, etl::size(result))();
    }

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        vectorized_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized sub assign

    /*!
     * \copydoc sub_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    void sub_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2 || local_context().serial){
            vectorized_sub_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        pre_assign(expr);
        post_assign_compound(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignSub<detail::select_vector_mode<E, R>(), R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignSub<detail::select_vector_mode<E, R>(), R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Standard Mul Assign

    /*!
     * \brief Multiply the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] *= expr[i];
        }
    }

    //Direct Mul Assign

    template <typename E, typename R>
    void direct_mul_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignMul<value_t<R>,E>(m, expr, 0, size)();
    }

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        direct_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel direct mul assign

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2 || local_context().serial){
            direct_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        pre_assign(expr);
        post_assign_compound(expr);

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

    //Vectorized Mul Assign

    template <typename E, typename R>
    void vectorized_mul_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        detail::VectorizedAssignMul<detail::select_vector_mode<E, R>(), R, E>(result, expr, 0, etl::size(result))();
    }

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        vectorized_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized mul assign

    /*!
     * \copydoc mul_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    void mul_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2 || local_context().serial){
            vectorized_mul_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        pre_assign(expr);
        post_assign_compound(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignMul<detail::select_vector_mode<E, R>(), R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignMul<detail::select_vector_mode<E, R>(), R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Standard Div Assign

    /*!
     * \brief Divide the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_enable_if(detail::standard_compound<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] /= expr[i];
        }
    }

    //Direct Div Assign

    template <typename E, typename R>
    void direct_div_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignDiv<value_t<R>,E>(m, expr, 0, size)();
    }

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::direct_compound<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        direct_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel direct Div assign

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_compound<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        const auto n = etl::size(result);

        if(n < parallel_threshold || threads < 2 || local_context().serial){
            direct_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        pre_assign(expr);
        post_assign_compound(expr);

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

    //Vectorized Div Assign

    template <typename E, typename R>
    void vectorized_div_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        detail::VectorizedAssignDiv<detail::select_vector_mode<E, R>(), R, E>(result, expr, 0, etl::size(result))();
    }

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::vectorized_compound<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        vectorized_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel vectorized div assign

    /*!
     * \copydoc div_evaluate
     */
    template <typename E, typename R, cpp_enable_if(detail::parallel_vectorized_compound<E, R>::value)>
    void div_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2 || local_context().serial){
            vectorized_div_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        pre_assign(expr);
        post_assign_compound(expr);

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssignDiv<detail::select_vector_mode<E, R>(), R, E>(result, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssignDiv<detail::select_vector_mode<E, R>(), R, E>(result, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Standard Mod Evaluate (no optimized versions for mod)

    /*!
     * \brief Modulo the result by the result of the expression expression
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R>
    void mod_evaluate(E&& expr, R&& result) {
        pre_assign(expr);
        post_assign_compound(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] %= expr[i];
        }
    }

    //Note: In case of direct evaluation, the temporary_expr itself must
    //not beevaluated by the static_visitor, otherwise, the result would
    //be evaluated twice and a temporary would be allocated for nothing

    /*!
     * \brief Assign the result of the expression expression to the result
     * \param expr The right hand side expression
     * \param result The left hand side
     */
    template <typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    void assign_evaluate(E&& expr, R&& result) {
        //Evaluate sub parts, if any
        pre_assign(expr);

        constexpr bool linear = decay_traits<E>::is_linear;

        if(!linear && result.alias(expr)){
            auto tmp_result = force_temporary(result);

            //Perform the evaluation to tmp_result
            assign_evaluate_impl(expr, tmp_result);

            //Perform the real evaluation to result
            assign_evaluate_impl(tmp_result, result);
        } else {
            //Perform the real evaluation, selected by TMP
            assign_evaluate_impl(expr, result);
        }

        post_assign(expr, result);
    }

    /*!
     * \copydoc assign_evaluate
     */
    template <typename E, typename R, cpp_enable_if(is_temporary_unary_expr<E>::value)>
    void assign_evaluate(E&& expr, R&& result) {
        pre_assign(expr.a());

        expr.direct_evaluate(result);

        post_assign(expr, result);
    }

    /*!
     * \copydoc assign_evaluate
     */
    template <typename E, typename R, cpp_enable_if(is_temporary_binary_expr<E>::value)>
    void assign_evaluate(E&& expr, R&& result) {
        pre_assign(expr.a());
        pre_assign(expr.b());

        expr.direct_evaluate(result);

        post_assign(expr, result);
    }

} // end of namespace standard_evaluator

//Only containers of the same storage order can be assigned directly
//Generators can be assigned to everything
template <typename Expr, typename Result>
struct direct_assign_compatible : cpp::or_u<
                                      decay_traits<Expr>::is_generator,
                                      decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order> {};

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!detail::has_optimized_evaluation<Expr, Result>::value, direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!detail::has_optimized_evaluation<Expr, Result>::value, !direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::assign_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(detail::is_direct_transpose<Expr, Result>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    // Make sure we have the data in CPU
    standard_evaluator::pre_assign(expr);
    standard_evaluator::post_assign_force(expr);

    // Perform transpose in memory
    detail::transpose<typename std::decay_t<Expr>::expr_t::sub_type, Result>::apply(expr.value().value(), std::forward<Result>(result));
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          assign_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    assign_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_parallel_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    auto old_parallel = local_context().parallel;

    local_context().parallel = true;

    assign_evaluate(expr.value(), std::forward<Result>(result));

    local_context().parallel = old_parallel;
}

/*!
 * \brief Evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void assign_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    assign_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::add_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          add_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    add_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Compound add evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void add_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    add_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(+=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound subtract evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::sub_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound sub evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          sub_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Compound sub evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    sub_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Compound sub evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void sub_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    sub_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(-=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound multiply evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mul_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound mul evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          mul_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Compound mul evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    mul_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Compound mul evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void mul_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    mul_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(*=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound divide evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::div_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound div evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          div_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Compound div evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    div_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Compound div evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void div_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    div_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(/=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mod_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound modulo evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(!direct_assign_compatible<Expr, Result>::value, !is_wrapper_expr<Expr>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    standard_evaluator::mod_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Compound mod evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_optimized_expr<Expr>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    optimized_forward(expr.value(),
                      [&result](auto& optimized) {
                          mod_evaluate(optimized, std::forward<Result>(result));
                      });
}

/*!
 * \brief Compound mod evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_serial_expr<Expr>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    auto old_serial = local_context().serial;

    local_context().serial = false;

    mod_evaluate(expr.value(), std::forward<Result>(result));

    local_context().serial = old_serial;
}

/*!
 * \brief Compound mod evaluation of the expr into result
 * \param expr The right hand side expression
 * \param result The left hand side
 */
template <typename Expr, typename Result, cpp_enable_if(is_timed_expr<Expr>::value)>
void mod_evaluate(Expr&& expr, Result&& result) {
    using resolution = typename std::decay_t<Expr>::clock_resolution;

    auto start_time = etl::timer_clock::now();

    mod_evaluate(expr.value(), std::forward<Result>(result));

    auto end_time = etl::timer_clock::now();
    auto duration = std::chrono::duration_cast<resolution>(end_time - start_time);

    std::cout << "timed(%=): " << expr.value() << " took " << duration.count() << resolution_to_string<resolution>() << std::endl;
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
    standard_evaluator::pre_assign(std::forward<Expr>(expr));
    standard_evaluator::post_assign_force(std::forward<Expr>(expr));
}

} //end of namespace etl
