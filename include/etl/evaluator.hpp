//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/parallel.hpp"

#include "etl/traits_lite.hpp" //forward declaration of the traits
#include "etl/visitor.hpp"     //visitor of the expressions
#include "etl/threshold.hpp"    //parallel thresholds

namespace etl {

namespace detail {

struct temporary_allocator_static_visitor : etl_visitor<temporary_allocator_static_visitor> {
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_temporary_visitor>;

    using etl_visitor<temporary_allocator_static_visitor>::operator();

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        v.allocate_temporary();

        (*this)(v.a());
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        v.allocate_temporary();

        (*this)(v.a());
        (*this)(v.b());
    }
};

struct evaluator_static_visitor : etl_visitor<evaluator_static_visitor> {
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_evaluator_visitor>;

    using etl_visitor<evaluator_static_visitor>::operator();

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        (*this)(v.a());

        v.evaluate();
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        (*this)(v.a());
        (*this)(v.b());

        v.evaluate();
    }
};

template<typename V_T, typename V_Expr>
struct Assign {
    mutable V_T* lhs;
    V_Expr& rhs;
    const std::size_t _first;
    const std::size_t _last;
    const std::size_t _size;

    Assign(V_T* lhs, V_Expr& rhs, std::size_t first, std::size_t last)
            : lhs(lhs), rhs(rhs), _first(first), _last(last), _size(last - first) {
        //Nothing else
    }

    void operator()() const {
        std::size_t iend = _first;

        if (unroll_normal_loops) {
            iend = _first + (_size & std::size_t(-4));

            for (std::size_t i = _first; i < iend; i += 4) {
                lhs[i]     = rhs[i];
                lhs[i + 1] = rhs[i + 1];
                lhs[i + 2] = rhs[i + 2];
                lhs[i + 3] = rhs[i + 3];
            }
        }

        for (std::size_t i = iend; i < _last; ++i) {
            lhs[i] = rhs[i];
        }
    }
};

template<typename V_T, typename V_Expr>
struct VectorizedAssign {
    mutable V_T* lhs;
    V_Expr& rhs;
    const std::size_t _first;
    const std::size_t _last;
    const std::size_t _size;

    using IT = intrinsic_traits<value_t<V_Expr>>;

    VectorizedAssign(V_T* lhs, V_Expr& rhs, std::size_t first, std::size_t last)
            : lhs(lhs), rhs(rhs), _first(first), _last(last), _size(last - first) {
        //Nothing else
    }

    std::size_t peel_loop() const {
        std::size_t i = 0;

        constexpr const auto size_1 = sizeof(value_t<V_Expr>);
        auto u_bytes                = (reinterpret_cast<uintptr_t>(lhs + _first) % IT::alignment);

        if (u_bytes >= size_1 && u_bytes % size_1 == 0) {
            auto u_loads = std::min(u_bytes / size_1, _size);

            for (; i < u_loads; ++i) {
                lhs[_first + i] = rhs[_first + i];
            }
        }

        return i;
    }

    inline std::size_t aligned_main_loop(std::size_t first) const {
        std::size_t i = 0;

        if(unroll_vectorized_loops && _last - first > IT::size * 4){
            for(i = first; i + IT::size * 4 - 1 < _last; i += IT::size * 4){
                vec::store(lhs + i, rhs.load(i));
                vec::store(lhs + i + 1 * IT::size, rhs.load(i + 1 * IT::size));
                vec::store(lhs + i + 2 * IT::size, rhs.load(i + 2 * IT::size));
                vec::store(lhs + i + 3 * IT::size, rhs.load(i + 3 * IT::size));
            }
        } else {
            for(i = first; i + IT::size - 1 < _last; i += IT::size){
                vec::store(lhs + i, rhs.load(i));
            }
        }

        return i;
    }

    inline std::size_t unaligned_main_loop(std::size_t first) const {
        std::size_t i;

        if(unroll_vectorized_loops && _last - first > IT::size * 4){
            for(i = first; i + IT::size * 4 - 1 < _last; i += IT::size * 4){
                vec::storeu(lhs + i, rhs.load(i));
                vec::storeu(lhs + i + 1 * IT::size, rhs.load(i + 1 * IT::size));
                vec::storeu(lhs + i + 2 * IT::size, rhs.load(i + 2 * IT::size));
                vec::storeu(lhs + i + 3 * IT::size, rhs.load(i + 3 * IT::size));
            }
        } else {
            for(i = first; i + IT::size - 1 < _last; i += IT::size){
                vec::storeu(lhs + i, rhs.load(i));
            }
        }

        return i;
    }

    void remainder_loop(std::size_t first) const {
        for (std::size_t i = first; i < _last; ++i) {
            lhs[i] = rhs[i];
        }
    }

    void operator()() const {
        //1. Peel loop (if necessary)
        auto peeled = peel_loop();

        //2. Main vectorized loop

        std::size_t first = peeled;

        if (_size - peeled >= IT::size) {
            if (reinterpret_cast<uintptr_t>(lhs + _first + peeled) % IT::alignment == 0) {
                first = aligned_main_loop(_first + peeled);
            } else {
                first = unaligned_main_loop(_first + peeled);
            }
        }

        //3. Remainder loop (non-vectorized)

        remainder_loop(first);
    }
};

template<typename V_T, typename V_Expr>
struct AssignAdd {
    mutable V_T* lhs;
    V_Expr& rhs;
    const std::size_t _first;
    const std::size_t _last;
    const std::size_t _size;

    AssignAdd(V_T* lhs, V_Expr& rhs, std::size_t first, std::size_t last)
            : lhs(lhs), rhs(rhs), _first(first), _last(last), _size(last - first) {
        //Nothing else
    }

    void operator()() const {
        std::size_t iend = _first;

        if (unroll_normal_loops) {
            iend = _first + (_size & std::size_t(-4));

            for (std::size_t i = _first; i < iend; i += 4) {
                lhs[i]     += rhs[i];
                lhs[i + 1] += rhs[i + 1];
                lhs[i + 2] += rhs[i + 2];
                lhs[i + 3] += rhs[i + 3];
            }
        }

        for (std::size_t i = iend; i < _last; ++i) {
            lhs[i] += rhs[i];
        }
    }
};

} //end of namespace detail

template <typename Expr, typename Result>
struct standard_evaluator {
    template <typename E>
    static void evaluate_only(E&& expr) {
        apply_visitor<detail::temporary_allocator_static_visitor>(expr);
        apply_visitor<detail::evaluator_static_visitor>(expr);
    }

    template <typename E, typename R>
    struct fast_assign : cpp::and_u<
                             has_direct_access<E>::value,
                             has_direct_access<R>::value,
                             !is_temporary_expr<E>::value> {};

    template <typename E, typename R>
    struct parallel_vectorized_assign : cpp::and_u<
                                   !fast_assign<E, R>::value,
                                   vectorize_expr,
                                   parallel,
                                   decay_traits<E>::vectorizable,
                                   intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                                   !is_temporary_expr<E>::value,
                                   std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

    template <typename E, typename R>
    struct vectorized_assign : cpp::and_u<
                                   !fast_assign<E, R>::value,
                                   !parallel_vectorized_assign<E, R>::value,
                                   vectorize_expr,
                                   decay_traits<E>::vectorizable,
                                   intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                                   !is_temporary_expr<E>::value,
                                   std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

    template <typename E, typename R>
    struct parallel_assign : cpp::and_u<
                                   has_direct_access<R>::value,
                                   !fast_assign<E, R>::value,
                                   !parallel_vectorized_assign<E, R>::value,
                                   parallel,
                                   !is_temporary_expr<E>::value> {};

    template <typename E, typename R>
    struct direct_assign : cpp::and_u<
                               !fast_assign<E, R>::value,
                               !parallel_assign<E, R>::value,
                               !parallel_vectorized_assign<E, R>::value,
                               !vectorized_assign<E, R>::value,
                               has_direct_access<R>::value,
                               !is_temporary_expr<E>::value> {};

    template <typename E, typename R>
    struct standard_assign : cpp::and_u<
                                 !fast_assign<E, R>::value,
                                 !parallel_assign<E, R>::value,
                                 !parallel_vectorized_assign<E, R>::value,
                                 !vectorized_assign<E, R>::value,
                                 !has_direct_access<R>::value,
                                 !is_temporary_expr<E>::value> {};

    //Standard assign version

    template <typename E, typename R, cpp_enable_if(standard_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] = expr[i];
        }
    }

    //Fast assign version (memory copy)

    template <typename E, typename R, cpp_enable_if(fast_assign<E, R>::value)>
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

    template <typename E, typename R, cpp_enable_if(direct_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        direct_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Parallel assign version

    template <typename E, typename R, cpp_enable_if(parallel_assign<E, R>::value)>
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

    template <typename E, typename R, cpp_enable_if(parallel_vectorized_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        static cpp::default_thread_pool<> pool(threads - 1);

        const std::size_t size = etl::size(result);

        if(size < parallel_threshold || threads < 2){
            vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
            return;
        }

        //Evaluate the sub parts of the expression, if any
        evaluate_only(expr);

        auto m = result.memory_start();

        auto batch = size / threads;

        //Schedule threads - 1 tasks
        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(detail::VectorizedAssign<value_t<R>, E>(m, expr, t * batch, (t+1) * batch));
        }

        //Perform the last task on the current threads
        detail::VectorizedAssign<value_t<R>, E>(m, expr, (threads - 1) * batch, size)();

        //Wait for the other threads
        pool.wait();
    }

    //Vectorized assign version

    template <typename E, typename R>
    static void vectorized_assign_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::VectorizedAssign<value_t<R>, E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void assign_evaluate(E&& expr, R&& result) {
        vectorized_assign_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    //Add Assign

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] += expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void direct_add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        detail::AssignAdd<value_t<R>,E>(m, expr, 0, size)();
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        direct_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void vectorized_add_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes                = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if (u_bytes >= size_1 && u_bytes % size_1 == 0) {
            auto u_loads = std::min(u_bytes & -size_1, size);

            for (; i < u_loads; ++i) {
                m[i] += expr[i];
            }
        }

        //2. Vectorized loop

        if (size - i >= IT::size) {
            if (reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0) {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::store(m + i, vec::add(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::add(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::add(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::add(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::store(m + i, vec::add(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::storeu(m + i, vec::add(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::add(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::add(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::add(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::storeu(m + i, vec::add(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for (; i < size; ++i) {
            m[i] += expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result) {
        vectorized_add_evaluate(std::forward<E>(expr), std::forward<R>(result));
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] -= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m                 = result.memory_start();

        std::size_t i = 0;

        if (unroll_normal_loops) {
            for (; i < (size & std::size_t(-4)); i += 4) {
                m[i] -= expr[i];
                m[i + 1] -= expr[i + 1];
                m[i + 2] -= expr[i + 2];
                m[i + 3] -= expr[i + 3];
            }
        }

        for (; i < size; ++i) {
            m[i] -= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes                = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if (u_bytes >= size_1 && u_bytes % size_1 == 0) {
            auto u_loads = std::min(u_bytes & -size_1, size);

            for (; i < u_loads; ++i) {
                m[i] -= expr[i];
            }
        }

        //2. Vectorized loop

        if (size - i >= IT::size) {
            if (reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0) {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::store(m + i, vec::sub(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::sub(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::sub(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::sub(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::store(m + i, vec::sub(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::storeu(m + i, vec::sub(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::sub(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::sub(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::sub(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::storeu(m + i, vec::sub(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for (; i < size; ++i) {
            m[i] -= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] *= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m                 = result.memory_start();

        std::size_t i = 0;

        if (unroll_normal_loops) {
            for (; i < (size & std::size_t(-4)); i += 4) {
                m[i] *= expr[i];
                m[i + 1] *= expr[i + 1];
                m[i + 2] *= expr[i + 2];
                m[i + 3] *= expr[i + 3];
            }
        }

        for (; i < size; ++i) {
            m[i] *= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes                = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if (u_bytes >= size_1 && u_bytes % size_1 == 0) {
            auto u_loads = std::min(u_bytes & -size_1, size);

            for (; i < u_loads; ++i) {
                m[i] *= expr[i];
            }
        }

        //2. Vectorized loop

        if (size - i >= IT::size) {
            if (reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0) {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::store(m + i, vec::mul(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::mul(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::mul(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::mul(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::store(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::storeu(m + i, vec::mul(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::mul(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::mul(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::mul(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::storeu(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for (; i < size; ++i) {
            m[i] *= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        for (std::size_t i = 0; i < etl::size(result); ++i) {
            result[i] /= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m                 = result.memory_start();

        std::size_t i = 0;

        if (unroll_normal_loops) {
            for (; i < (size & std::size_t(-4)); i += 4) {
                m[i] /= expr[i];
                m[i + 1] /= expr[i + 1];
                m[i + 2] /= expr[i + 2];
                m[i + 3] /= expr[i + 3];
            }
        }

        for (; i < size; ++i) {
            m[i] /= expr[i];
        }
    }

    template <typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result) {
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes                = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if (u_bytes >= size_1 && u_bytes % size_1 == 0) {
            auto u_loads = std::min(u_bytes & -size_1, size);

            for (; i < u_loads; ++i) {
                m[i] /= expr[i];
            }
        }

        //2. Vectorized loop

        if (size - i >= IT::size) {
            if (reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0) {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::store(m + i, vec::div(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::div(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::div(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::div(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::store(m + i, vec::div(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if (unroll_vectorized_loops && size - i > IT::size * 4) {
                    for (; i + IT::size * 4 - 1 < size; i += IT::size * 4) {
                        vec::storeu(m + i, vec::div(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::div(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::div(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::div(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for (; i + IT::size - 1 < size; i += IT::size) {
                        vec::storeu(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for (; i < size; ++i) {
            m[i] /= expr[i];
        }
    }

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
