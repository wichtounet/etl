//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

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
    static bool is([[maybe_unused]] const Expr& expr) {
        return false;
    }

    /*!
     * \brief Indicates if the given expression or one of its sub expressions is optimizable or not
     * \param expr The expression to test
     * \return true if the expression or one of its sub expressions is optimizable, false otherwise
     */
    static bool is_deep([[maybe_unused]] const Expr& expr) {
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
    static constexpr bool is(const etl::unary_expr<T, Expr, UnaryOp>& /*unused*/) {
        return std::is_same_v<UnaryOp, plus_unary_op<T>>;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        return is(expr) || is_optimizable_deep(expr.value);
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
    static constexpr bool is(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& /*unused*/) {
        if (std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            return true;
        }

        if (std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            return true;
        }

        if (std::is_same_v<BinaryOp, div_binary_op<T>>) {
            return true;
        }

        if (std::is_same_v<BinaryOp, minus_binary_op<T>>) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs) || is_optimizable_deep(expr.rhs);
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
        if (expr.lhs.value == 1.0 && std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            return true;
        }

        if (expr.lhs.value == 0.0 && std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            return true;
        }

        if (expr.lhs.value == 0.0 && std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            return true;
        }

        if (expr.lhs.value == 0.0 && std::is_same_v<BinaryOp, div_binary_op<T>>) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs) || is_optimizable_deep(expr.rhs);
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
        if (expr.rhs.value == 1.0 && std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            return true;
        }

        if (expr.rhs.value == 0.0 && std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            return true;
        }

        if (expr.rhs.value == 0.0 && std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            return true;
        }

        if (expr.rhs.value == 0.0 && std::is_same_v<BinaryOp, minus_binary_op<T>>) {
            return true;
        }

        if (expr.rhs.value == 1.0 && std::is_same_v<BinaryOp, div_binary_op<T>>) {
            return true;
        }

        return false;
    }

    /*! \copydoc optimizable::is_deep */
    static bool is_deep(const etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>& expr) {
        return is(expr) || is_optimizable_deep(expr.lhs) || is_optimizable_deep(expr.rhs);
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
        return is_optimizable_deep(expr.lhs) || is_optimizable_deep(expr.rhs);
    }
};

/*!
 * \brief Function to test if expr is optimizable
 * \param expr The expression to test
 * \return true if the expression is optimizable or not
 */
template <typename Expr>
bool is_optimizable(const Expr& expr) {
    return optimizable<std::decay_t<Expr>>::is(expr);
}

/*!
 * \brief Function to test if expr or sub parts of expr are optimizable
 * \param expr The expression to test
 * \return true if the expression is deeply optimizable or not
 */
template <typename Expr>
bool is_optimizable_deep(const Expr& expr) {
    return optimizable<std::decay_t<Expr>>::is_deep(expr);
}

/*!
 * \brief Transformer functor for optimizable expression
 */
template <typename Expr>
struct transformer {
    /*!
     * \brief Transform the expression using the given builder
     * \param builder The builder to use
     * \param expr The expression to transform
     */
    template <typename Builder>
    static void transform([[maybe_unused]] Builder builder, [[maybe_unused]] const Expr& expr) {
        std::cout << "Arrived in parent, should not happen" << std::endl;
    }
};

/*!
 * \copydoc transformer
 *
 * Specialization for unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct transformer<etl::unary_expr<T, Expr, UnaryOp>> {
    /*!
     * \brief Transform the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to transform
     */
    template <typename Builder>
    static void transform([[maybe_unused]] Builder parent_builder, [[maybe_unused]] const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        if constexpr (std::is_same_v<UnaryOp, plus_unary_op<T>>) {
            parent_builder(expr.value);
        }
    }
};

/*!
 * \copydoc transformer
 *
 * Specialization for binary_expr<Scalar, Scalar>
 */
template <typename T, typename BinaryOp>
struct transformer<etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>> {
    /*!
     * \brief Transform the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to transform
     */
    template <typename Builder>
    static void transform([[maybe_unused]] Builder parent_builder, [[maybe_unused]] const etl::binary_expr<T, etl::scalar<T>, BinaryOp, etl::scalar<T>>& expr) {
        if constexpr (std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            parent_builder(etl::scalar<T>(expr.lhs.value * expr.rhs.value));
        } else if constexpr (std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            parent_builder(etl::scalar<T>(expr.lhs.value + expr.rhs.value));
        } else if constexpr (std::is_same_v<BinaryOp, minus_binary_op<T>>) {
            parent_builder(etl::scalar<T>(expr.lhs.value - expr.rhs.value));
        } else if constexpr (std::is_same_v<BinaryOp, div_binary_op<T>>) {
            parent_builder(etl::scalar<T>(expr.lhs.value / expr.rhs.value));
        }
    }
};

/*!
 * \copydoc transformer
 *
 * Specialization for binary_expr<Scalar, ETL>
 */
template <typename T, typename BinaryOp, typename RightExpr>
struct transformer<etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>> {
    /*!
     * \brief Transform the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to transform
     */
    template <typename Builder>
    static void transform([[maybe_unused]] Builder parent_builder, [[maybe_unused]] const etl::binary_expr<T, etl::scalar<T>, BinaryOp, RightExpr>& expr) {
        if constexpr (std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            if (expr.lhs.value == 1.0) {
                parent_builder(expr.rhs);
            } else if (expr.lhs.value == 0.0) {
                parent_builder(expr.lhs);
            }
        } else if constexpr (std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            if (expr.lhs.value == 0.0) {
                parent_builder(expr.rhs);
            }
        } else if constexpr (std::is_same_v<BinaryOp, div_binary_op<T>>) {
            if (expr.lhs.value == 0.0) {
                parent_builder(expr.lhs);
            }
        }
    }
};

/*!
 * \copydoc transformer
 *
 * Specialization for binary_expr<ETL, Scalar>
 */
template <typename T, typename LeftExpr, typename BinaryOp>
struct transformer<etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>> {
    /*!
     * \brief Transform the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to transform
     */
    template <typename Builder>
    static void transform([[maybe_unused]] Builder parent_builder, [[maybe_unused]] const etl::binary_expr<T, LeftExpr, BinaryOp, etl::scalar<T>>& expr) {
        if constexpr (std::is_same_v<BinaryOp, mul_binary_op<T>>) {
            if (expr.rhs.value == 1.0) {
                parent_builder(expr.lhs);
            } else if (expr.rhs.value == 0.0) {
                parent_builder(expr.rhs);
            }
        } else if constexpr (std::is_same_v<BinaryOp, plus_binary_op<T>>) {
            if (expr.rhs.value == 0.0) {
                parent_builder(expr.lhs);
            }
        } else if constexpr (std::is_same_v<BinaryOp, minus_binary_op<T>>) {
            if (expr.rhs.value == 0.0) {
                parent_builder(expr.lhs);
            }
        } else if constexpr (std::is_same_v<BinaryOp, div_binary_op<T>>) {
            if (expr.rhs.value == 1.0) {
                parent_builder(expr.lhs);
            }
        }
    }
};

/*!
 * \brief Function to transform the expression into its optimized form
 * \param parent_builder The builder of its parent node
 * \param expr The expression to optimize
 */
template <typename Builder, typename Expr>
void transform(Builder parent_builder, const Expr& expr) {
    transformer<std::decay_t<Expr>>::transform(parent_builder, expr);
}

/*!
 * \brief An optimizer for the given expression type
 */
template <typename Expr>
struct optimizer {
    /*!
     * \brief Optimize the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to optimize
     */
    template <typename Builder>
    static void apply([[maybe_unused]] Builder parent_builder, [[maybe_unused]] const Expr& expr) {
        std::cout << "Leaf node" << std::endl;
    }
};

/*!
 * \brief An optimizer for unary expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct optimizer<etl::unary_expr<T, Expr, UnaryOp>> {
    /*!
     * \brief Optimize the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to optimize
     */
    template <typename Builder>
    static void apply(Builder parent_builder, const etl::unary_expr<T, Expr, UnaryOp>& expr) {
        if (is_optimizable(expr)) {
            transform(parent_builder, expr);
        } else if (is_optimizable_deep(expr.value)) {
            auto value_builder = [&](auto&& new_value) {
                parent_builder(etl::unary_expr<T, etl::detail::build_type<decltype(new_value)>, UnaryOp>(new_value));
            };

            optimize(value_builder, expr.value);
        } else {
            parent_builder(expr);
        }
    }
};

/*!
 * \brief An optimizer for binary expr
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct optimizer<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    /*!
     * \brief Optimize the expression using the given builder
     * \param parent_builder The builder to use
     * \param expr The expression to optimize
     */
    template <typename Builder>
    static void apply(Builder parent_builder, const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& expr) {
        if (is_optimizable(expr)) {
            transform(parent_builder, expr);
        } else if (is_optimizable_deep(expr.lhs)) {
            auto lhs_builder = [&](auto&& new_lhs) {
                parent_builder(etl::binary_expr<T, etl::detail::build_type<decltype(new_lhs)>, BinaryOp, RightExpr>(new_lhs, expr.rhs));
            };

            optimize(lhs_builder, expr.lhs);
        } else if (is_optimizable_deep(expr.rhs)) {
            auto rhs_builder = [&](auto&& new_rhs) {
                parent_builder(etl::binary_expr<T, LeftExpr, BinaryOp, etl::detail::build_type<decltype(new_rhs)>>(expr.lhs, new_rhs));
            };

            optimize(rhs_builder, expr.rhs);
        } else {
            parent_builder(expr);
        }
    }
};

/*!
 * \brief Optimize an expression and reconstruct the parent from the
 * optimized expression.
 * \param parent_builder The builder to rebuild the parent
 * \param expr The expression to optimize
 */
template <typename Builder, typename Expr>
void optimize(Builder parent_builder, Expr& expr) {
    optimizer<std::decay_t<Expr>>::apply(parent_builder, expr);
}

/*!
 * \brief Optimize an expression and pass the optimized expression
 * to the given functor
 * \param expr The expression to optimize
 * \param result The functor to apply on the optimized expression
 */
template <typename Expr, typename Result>
void optimized_forward(Expr& expr, Result result) {
    if (is_optimizable_deep(expr)) {
        optimize([result](auto&& new_expr) mutable { optimized_forward(new_expr, result); }, expr);
        return;
    }

    result(expr);
}

} //end of namespace etl
