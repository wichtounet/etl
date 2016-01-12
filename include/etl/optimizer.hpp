//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/visitor.hpp"     //forward declaration of the traits

namespace etl {

/*!
 * \brief Simple traits to test if an expression is optimizable
 *
 * The default traits implementation return false for each
 * expression. The traits should be specialized for each optimizable
 * expression.
 */
template <typename Expr>
struct optimizable {
    /*!
     * \brief Indicates if the given expression is optimizable or not
     * \param expr The expression to test
     * \return true if the expression is optimizable, false otherwise
     */
    static bool is(const Expr& expr) {
        cpp_unused(expr);
        return false;
    }

    /*!
     * \brief Indicates if the given expression or one of its sub expressions is optimizable or not
     * \param expr The expression to test
     * \return true if the expression or one of its sub expressions is optimizable, false otherwise
     */
    static bool is_deep(const Expr& expr) {
        cpp_unused(expr);
        return false;
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct optimizable<etl::unary_expr<T, Expr, UnaryOp>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::unary_expr<T, Expr, UnaryOp>& /*unused*/) {
        return std::is_same<UnaryOp, plus_unary_op<T>>::value;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        return is(expr) || is_optimizable_deep(expr.value());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for binary_expr and two scalars
 */
template <typename T, typename BinaryOp>
struct optimizable<etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& /*unused*/) {
        if (std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            return true;
        }

        if (std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            return true;
        }

        if (std::is_same<BinaryOp, div_binary_op<T>>::value) {
            return true;
        }

        if (std::is_same<BinaryOp, minus_binary_op<T>>::value) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs()) || is_optimizable_deep(expr.rhs());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for binary_expr with scalar lhs
 */
template <typename T, typename BinaryOp, typename RightExpr>
struct optimizable<etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>& expr) {
        if (expr.lhs().value == 1.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            return true;
        }

        if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            return true;
        }

        if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            return true;
        }

        if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, div_binary_op<T>>::value) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs()) || is_optimizable_deep(expr.rhs());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for binary_expr with scalar rhs
 */
template <typename T, typename LeftExpr, typename BinaryOp>
struct optimizable<etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>& expr) {
        if (expr.rhs().value == 1.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            return true;
        }

        if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            return true;
        }

        if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            return true;
        }

        if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, minus_binary_op<T>>::value) {
            return true;
        }

        if (expr.rhs().value == 1.0 && std::is_same<BinaryOp, div_binary_op<T>>::value) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs()) || is_optimizable_deep(expr.rhs());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for general binary_expr
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct optimizable<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& /*unused*/) {
        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& expr) {
        return is_optimizable_deep(expr.lhs()) || is_optimizable_deep(expr.rhs());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for temporary_unary_expr
 */
template <typename T, typename A, typename Op, typename Forced>
struct optimizable<etl::temporary_unary_expr<T, A, Op, Forced>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::temporary_unary_expr<T, A, Op, Forced>& /*unused*/) {
        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::temporary_unary_expr<T, A, Op, Forced>& expr) {
        return is_optimizable_deep(expr.a());
    }
};

/*!
 * \copydoc optimizable
 *
 * Specialization for temporary_binary_expr
 */
template <typename T, typename A, typename B, typename Op, typename Forced>
struct optimizable<etl::temporary_binary_expr<T, A, B, Op, Forced>> {
    /*! \copydoc optimizable::is */
    static bool is(const etl::temporary_binary_expr<T, A, B, Op, Forced>& /*unused*/) {
        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::temporary_binary_expr<T, A, B, Op, Forced>& expr) {
        return is_optimizable_deep(expr.a()) || is_optimizable_deep(expr.b());
    }
};

template <typename Expr>
bool is_optimizable(const Expr& expr) {
    return optimizable<std::decay_t<Expr>>::is(expr);
}

template <typename Expr>
bool is_optimizable_deep(const Expr& expr) {
    return optimizable<std::decay_t<Expr>>::is_deep(expr);
}

template <typename Expr>
struct transformer {
    template <typename Builder>
    static void transform(Builder /*unused*/, const Expr& /*unused*/) {
        std::cout << "Arrived in parent, should not happen" << std::endl;
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct transformer<etl::unary_expr<T, Expr, UnaryOp>> {
    template <typename Builder>
    static void transform(Builder parent_builder, const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        if (std::is_same<UnaryOp, plus_unary_op<T>>::value) {
            parent_builder(expr.value());
        }
    }
};

template <typename T, typename BinaryOp>
struct transformer<etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>> {
    template <typename Builder>
    static void transform(Builder parent_builder, const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& expr) {
        if (std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            parent_builder(etl::scalar<T>(expr.lhs().value * expr.rhs().value));
        } else if (std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            parent_builder(etl::scalar<T>(expr.lhs().value + expr.rhs().value));
        } else if (std::is_same<BinaryOp, minus_binary_op<T>>::value) {
            parent_builder(etl::scalar<T>(expr.lhs().value - expr.rhs().value));
        } else if (std::is_same<BinaryOp, div_binary_op<T>>::value) {
            parent_builder(etl::scalar<T>(expr.lhs().value / expr.rhs().value));
        }
    }
};

template <typename T, typename BinaryOp, typename RightExpr>
struct transformer<etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>> {
    template <typename Builder>
    static void transform(Builder parent_builder, const etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>& expr) {
        if (expr.lhs().value == 1.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            parent_builder(expr.rhs());
        } else if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        } else if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            parent_builder(expr.rhs());
        } else if (expr.lhs().value == 0.0 && std::is_same<BinaryOp, div_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        }
    }
};

template <typename T, typename LeftExpr, typename BinaryOp>
struct transformer<etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>> {
    template <typename Builder>
    static void transform(Builder parent_builder, const etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>& expr) {
        if (expr.rhs().value == 1.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        } else if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, mul_binary_op<T>>::value) {
            parent_builder(expr.rhs());
        } else if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, plus_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        } else if (expr.rhs().value == 0.0 && std::is_same<BinaryOp, minus_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        } else if (expr.rhs().value == 1.0 && std::is_same<BinaryOp, div_binary_op<T>>::value) {
            parent_builder(expr.lhs());
        }
    }
};

template <typename Builder, typename Expr>
void transform(Builder parent_builder, const Expr& expr) {
    transformer<std::decay_t<Expr>>::transform(parent_builder, expr);
}

template <typename Expr>
struct optimizer {
    template <typename Builder>
    static void apply(Builder /*unused*/, const Expr& /*unused*/) {
        std::cout << "Leaf node" << std::endl;
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct optimizer<etl::unary_expr<T, Expr, UnaryOp>> {
    template <typename Builder>
    static void apply(Builder parent_builder, const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        if (is_optimizable(expr)) {
            transform(parent_builder, expr);
        } else if (is_optimizable_deep(expr.value())) {
            auto value_builder = [&](const auto& new_value) {
                parent_builder(etl::unary_expr<T, etl::detail::build_type<decltype(new_value)>, UnaryOp>(new_value));
            };

            optimize(value_builder, expr.value());
        } else {
            parent_builder(expr);
        }
    }
};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct optimizer<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    template <typename Builder>
    static void apply(Builder parent_builder, const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& expr) {
        if (is_optimizable(expr)) {
            transform(parent_builder, expr);
        } else if (is_optimizable_deep(expr.lhs())) {
            auto lhs_builder = [&](const auto& new_lhs) {
                parent_builder(etl::binary_expr<T, etl::detail::build_type<decltype(new_lhs)>, BinaryOp, RightExpr>(new_lhs, expr.rhs()));
            };

            optimize(lhs_builder, expr.lhs());
        } else if (is_optimizable_deep(expr.rhs())) {
            auto rhs_builder = [&](const auto& new_rhs) {
                parent_builder(etl::binary_expr<T, LeftExpr, BinaryOp, etl::detail::build_type<decltype(new_rhs)>>(expr.lhs(), new_rhs));
            };

            optimize(rhs_builder, expr.rhs());
        } else {
            parent_builder(expr);
        }
    }
};

template <typename T, typename A, typename Op, typename Forced>
struct optimizer<etl::temporary_unary_expr<T, A, Op, Forced>> {
    template <typename Builder>
    static void is(Builder parent_builder, const etl::temporary_unary_expr<T, A, Op, Forced>& expr) {
        if (is_optimizable_deep(expr.a())) {
            auto lhs_builder = [&](const auto& new_lhs) {
                parent_builder(etl::temporary_unary_expr<T, etl::detail::build_type<decltype(new_lhs)>, Op, Forced>(new_lhs));
            };

            optimize(lhs_builder, expr.a());
        } else {
            parent_builder(expr);
        }
    }
};

template <typename T, typename A, typename B, typename Op, typename Forced>
struct optimizer<etl::temporary_binary_expr<T, A, B, Op, Forced>> {
    template <typename Builder>
    static void apply(Builder parent_builder, const etl::temporary_binary_expr<T, A, B, Op, Forced>& expr) {
        if (is_optimizable_deep(expr.a())) {
            auto lhs_builder = [&](const auto& new_lhs) {
                parent_builder(etl::temporary_binary_expr<T, etl::detail::build_type<decltype(new_lhs)>, B, Op, Forced>(new_lhs));
            };

            optimize(lhs_builder, expr.a());
        } else if (is_optimizable_deep(expr.b())) {
            auto rhs_builder = [&](const auto& new_rhs) {
                parent_builder(etl::temporary_binary_expr<T, A, etl::detail::build_type<decltype(new_rhs)>, Op, Forced>(new_rhs));
            };

            optimize(rhs_builder, expr.b());
        } else {
            parent_builder(expr);
        }
    }
};

template <typename Builder, typename Expr>
void optimize(Builder parent_builder, const Expr& expr) {
    optimizer<std::decay_t<Expr>>::apply(parent_builder, expr);
}

template <typename Expr, typename Result>
void optimized_forward(const Expr& expr, Result result) {
    if (is_optimizable_deep(expr)) {
        optimize([result](const auto& new_expr) { optimized_forward(new_expr, result); }, expr);
        return;
    }

    result(expr);
}

} //end of namespace etl
