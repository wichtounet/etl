//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_EVALUATOR_HPP
#define ETL_EVALUATOR_HPP

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
    struct vectorized_assign : cpp::and_u<vectorize_expr, decay_traits<E>::vectorizable, intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable> {};

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
    
    template<typename E, typename R, cpp_enable_if(!vectorized_assign<E, R>::value && !has_direct_access<R>::value && !is_temporary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        const std::size_t size = etl::size(result);

        for(std::size_t i = 0; i < size; ++i){
            result[i] = expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(vectorized_assign<E, R>::value && !is_temporary_expr<E>::value)>
    static void assign_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        using IT = intrinsic_traits<value_t<E>>;

        auto m = result.memory_start();

        const std::size_t size = etl::size(result);

        std::size_t iend = 0;

        if(unroll_vectorized_loops){
            iend = size & std::size_t(-IT::size * 4);

            for(std::size_t i = 0; i < iend; i += IT::size * 4){
                vec::store(m + i, expr.load(i));
                vec::store(m + i + 1 * IT::size, expr.load(i + 1 * IT::size));
                vec::store(m + i + 2 * IT::size, expr.load(i + 2 * IT::size));
                vec::store(m + i + 3 * IT::size, expr.load(i + 3 * IT::size));
            }
        } else {
            iend = size & std::size_t(-IT::size);

            for(std::size_t i = 0; i < iend; i += IT::size){
                vec::store(m + i, expr.load(i));
            }
        }

        //Finish the iterations in a non-vectorized fashion
        for(std::size_t i = iend; i < size; ++i){
            m[i] = expr[i];
        }
    }

    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    static void add_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] += expr[i];
        }
    }

    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    static void sub_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] -= expr[i];
        }
    }

    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    static void mul_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] *= expr[i];
        }
    }

    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    static void div_evaluate(E&& expr, R&& result){
        evaluate_only(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] /= expr[i];
        }
    }

    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
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

template<typename Expr, typename Result>
void assign_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::assign_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result>
void add_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::add_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result>
void sub_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::sub_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result>
void mul_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mul_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result>
void div_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::div_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr, typename Result>
void mod_evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::mod_evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

/*!
 * \brief Force the internal evaluation of an expression
 */
template<typename Expr>
void force(Expr&& expr){
    standard_evaluator<Expr, void>::evaluate_only(std::forward<Expr>(expr));
}

} //end of namespace etl

#endif
