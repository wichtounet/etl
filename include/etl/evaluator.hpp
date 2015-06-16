//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "traits_lite.hpp"   //forward declaration of the traits
#include "visitor.hpp"   //forward declaration of the traits

namespace etl {

namespace detail {

struct temporary_allocator_static_visitor : etl_visitor<temporary_allocator_static_visitor> {
    template<typename E>
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
    template<typename E>
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

} //end of namespace detail

template<typename Expr, typename Result>
struct standard_evaluator {
    template<typename E>
    static void evaluate_only(E&& expr){
        apply_visitor<detail::temporary_allocator_static_visitor>(expr);
        apply_visitor<detail::evaluator_static_visitor>(expr);
    }

    template<typename E, typename R>
    struct vectorized_assign : cpp::and_u<
        vectorize_expr,
        decay_traits<E>::vectorizable,
        intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
        std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

    //Standard assign version

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value && !is_temporary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] = expr[i];
        }
    }

    //Direct assign version

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value && !is_temporary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t iend = 0;

        if(unroll_normal_loops){
            iend = size & std::size_t(-4);

            for(std::size_t i = 0; i < iend; i += 4){
                m[i] = expr[i];
                m[i+1] = expr[i+1];
                m[i+2] = expr[i+2];
                m[i+3] = expr[i+3];
            }
        }

        for(std::size_t i = iend; i < size; ++i){
            m[i] = expr[i];
        }
    }

    //Vectorized assign version

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value && !is_temporary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if(u_bytes >= size_1 && u_bytes % size_1 == 0){
            auto u_loads = std::min(u_bytes & -size_1, size);

            for(; i < u_loads; ++i){
                m[i] = expr[i];
            }
        }

        //2. Vectorized loop

        if(size - i >= IT::size){
            if(reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0){
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::store(m + i,                   expr.load(i));
                        vec::store(m + i + 1 * IT::size,    expr.load(i + 1 * IT::size));
                        vec::store(m + i + 2 * IT::size,    expr.load(i + 2 * IT::size));
                        vec::store(m + i + 3 * IT::size,    expr.load(i + 3 * IT::size));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::store(m + i, expr.load(i));
                    }
                }
            } else {
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::storeu(m + i,                  expr.load(i));
                        vec::storeu(m + i + 1 * IT::size,   expr.load(i + 1 * IT::size));
                        vec::storeu(m + i + 2 * IT::size,   expr.load(i + 2 * IT::size));
                        vec::storeu(m + i + 3 * IT::size,   expr.load(i + 3 * IT::size));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::storeu(m + i, expr.load(i));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for(; i < size; ++i){
            m[i] = expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void add_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] += expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void add_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m = result.memory_start();

        std::size_t i = 0;

        if(unroll_normal_loops){
            for(; i < (size & std::size_t(-4)); i += 4){
                m[i] += expr[i];
                m[i+1] += expr[i+1];
                m[i+2] += expr[i+2];
                m[i+3] += expr[i+3];
            }
        }

        for(; i < size; ++i){
            m[i] += expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void add_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if(u_bytes >= size_1 && u_bytes % size_1 == 0){
            auto u_loads = std::min(u_bytes & -size_1, size);

            for(; i < u_loads; ++i){
                m[i] += expr[i];
            }
        }

        //2. Vectorized loop

        if(size - i >= IT::size){
            if(reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0){
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::store(m + i,                vec::add(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::add(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::add(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::add(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::store(m + i, vec::add(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::storeu(m + i,                vec::add(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::add(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::add(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::add(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::storeu(m + i, vec::add(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for(; i < size; ++i){
            m[i] += expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void sub_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] -= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void sub_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m = result.memory_start();

        std::size_t i = 0;

        if(unroll_normal_loops){
            for(; i < (size & std::size_t(-4)); i += 4){
                m[i] -= expr[i];
                m[i+1] -= expr[i+1];
                m[i+2] -= expr[i+2];
                m[i+3] -= expr[i+3];
            }
        }

        for(; i < size; ++i){
            m[i] -= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void sub_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if(u_bytes >= size_1 && u_bytes % size_1 == 0){
            auto u_loads = std::min(u_bytes & -size_1, size);

            for(; i < u_loads; ++i){
                m[i] -= expr[i];
            }
        }

        //2. Vectorized loop

        if(size - i >= IT::size){
            if(reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0){
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::store(m + i,                vec::sub(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::sub(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::sub(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::sub(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::store(m + i, vec::sub(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::storeu(m + i,                vec::sub(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::sub(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::sub(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::sub(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::storeu(m + i, vec::sub(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for(; i < size; ++i){
            m[i] -= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void mul_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] *= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void mul_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m = result.memory_start();

        std::size_t i = 0;

        if(unroll_normal_loops){
            for(; i < (size & std::size_t(-4)); i += 4){
                m[i] *= expr[i];
                m[i+1] *= expr[i+1];
                m[i+2] *= expr[i+2];
                m[i+3] *= expr[i+3];
            }
        }

        for(; i < size; ++i){
            m[i] *= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void mul_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if(u_bytes >= size_1 && u_bytes % size_1 == 0){
            auto u_loads = std::min(u_bytes & -size_1, size);

            for(; i < u_loads; ++i){
                m[i] *= expr[i];
            }
        }

        //2. Vectorized loop

        if(size - i >= IT::size){
            if(reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0){
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::store(m + i,                vec::mul(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::mul(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::mul(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::mul(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::store(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::storeu(m + i,                vec::mul(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::mul(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::mul(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::mul(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::storeu(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for(; i < size; ++i){
            m[i] *= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value)>
    static void div_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] /= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && has_direct_access<R>::value)>
    static void div_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        const std::size_t size = etl::size(result);
        auto m = result.memory_start();

        std::size_t i = 0;

        if(unroll_normal_loops){
            for(; i < (size & std::size_t(-4)); i += 4){
                m[i] /= expr[i];
                m[i+1] /= expr[i+1];
                m[i+2] /= expr[i+2];
                m[i+3] /= expr[i+3];
            }
        }

        for(; i < size; ++i){
            m[i] /= expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value)>
    static void div_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t i = 0;

        //1. Peel loop

        constexpr const auto size_1 = sizeof(value_t<E>);
        auto u_bytes = (reinterpret_cast<uintptr_t>(m) % IT::alignment);

        if(u_bytes >= size_1 && u_bytes % size_1 == 0){
            auto u_loads = std::min(u_bytes & -size_1, size);

            for(; i < u_loads; ++i){
                m[i] /= expr[i];
            }
        }

        //2. Vectorized loop

        if(size - i >= IT::size){
            if(reinterpret_cast<uintptr_t>(m + i) % IT::alignment == 0){
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::store(m + i,                vec::div(result.load(i), expr.load(i)));
                        vec::store(m + i + 1 * IT::size, vec::div(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::store(m + i + 2 * IT::size, vec::div(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::store(m + i + 3 * IT::size, vec::div(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::store(m + i, vec::div(result.load(i), expr.load(i)));
                    }
                }
            } else {
                if(unroll_vectorized_loops && size - i > IT::size * 4){
                    for(; i + IT::size * 4 - 1 < size; i += IT::size * 4){
                        vec::storeu(m + i,                vec::div(result.load(i), expr.load(i)));
                        vec::storeu(m + i + 1 * IT::size, vec::div(result.load(i + 1 * IT::size), expr.load(i + 1 * IT::size)));
                        vec::storeu(m + i + 2 * IT::size, vec::div(result.load(i + 2 * IT::size), expr.load(i + 2 * IT::size)));
                        vec::storeu(m + i + 3 * IT::size, vec::div(result.load(i + 3 * IT::size), expr.load(i + 3 * IT::size)));
                    }
                } else {
                    for(; i + IT::size  - 1 < size; i += IT::size){
                        vec::storeu(m + i, vec::mul(result.load(i), expr.load(i)));
                    }
                }
            }
        }

        //3. Remainder loop

        //Finish the iterations in a non-vectorized fashion
        for(; i < size; ++i){
            m[i] /= expr[i];
        }
    }

    template<typename E, typename R>
    static void mod_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] %= expr[i];
        }
    }

    //Note: In case of direct evaluation, the temporary_expr itself must
    //not beevaluated by the static_visitor, otherwise, the result would
    //be evaluated twice and a temporary would be allocated for nothing

    template<typename E, typename R, cpp_enable_if(is_temporary_unary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.a());
        apply_visitor<detail::evaluator_static_visitor>(expr.a());

        expr.direct_evaluate(result);
    }

    template<typename E, typename R, cpp_enable_if(is_temporary_binary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.a());
        apply_visitor<detail::temporary_allocator_static_visitor>(expr.b());

        apply_visitor<detail::evaluator_static_visitor>(expr.a());
        apply_visitor<detail::evaluator_static_visitor>(expr.b());

        expr.direct_evaluate(result);
    }
};

//Only containers of the same storage order can be assigned directly
//Generators can be assigned to everything
template<typename Expr, typename Result>
struct direct_assign_compatible : cpp::or_u<
    decay_traits<Expr>::is_generator,
    decay_traits<Expr>::storage_order == decay_traits<Result>::storage_order> {};

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void assign_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::assign_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void assign_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::assign_evaluate(transpose(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void add_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::add_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void add_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::add_evaluate(transpose(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void sub_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::sub_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void sub_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::sub_evaluate(transpose(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void mul_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mul_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void mul_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mul_evaluate(transpose(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void div_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::div_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void div_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::div_evaluate(transpose(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_enable_if(direct_assign_compatible<Expr, Result>::value)>
void mod_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mod_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result, cpp_disable_if(direct_assign_compatible<Expr, Result>::value)>
void mod_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mod_evaluate(transpose(expr), std::forward<Result>(result));
}

/*!
 * \brief Force the internal evaluation of an expression
 * \param expr The expression to force inner evaluation
 *
 * This function can be used when complex expressions are used
 * lazily.
 */
template<typename Expr>
void force(Expr&& expr){
    standard_evaluator<Expr, void>::evaluate_only(std::forward<Expr>(expr));
}

} //end of namespace etl
