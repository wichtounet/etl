//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_FAST_CATCH

inline void evaluate_result_direct(const char* file, size_t line, const char* exp, bool value){
#ifdef ETL_VERY_FAST_CATCH
    if (!value) {
#endif
        Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
        result.setResultType(value);
        result.setLhs(value ? "true" : "false");
        result.setOp("");
        result.endExpression();
        result.react();
#ifdef ETL_VERY_FAST_CATCH
    }
#endif
}

template<typename L, typename R>
void evaluate_result(const char* file, size_t line, const char* exp, L lhs, R rhs){
    bool bool_result = lhs == rhs;

#ifdef ETL_VERY_FAST_CATCH
    if (!bool_result) {
#endif
        Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
        result.setResultType(bool_result);

        // Note: This will break the display is successfull tests are
        // displayed
        if (!bool_result) {
            result.setLhs(Catch::toString(lhs));
            result.setRhs(Catch::toString(rhs));
            result.setOp("==");
        }

        result.endExpression();
        result.react();
#ifdef ETL_VERY_FAST_CATCH
    }
#endif
}

template<typename L>
void evaluate_result_approx(const char* file, size_t line, const char* exp, L lhs, decltype(lhs) rhs, double eps = base_eps){
    bool bool_result = std::abs(lhs - rhs) < eps * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));

#ifdef ETL_VERY_FAST_CATCH
    if (!bool_result) {
#endif
        Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
        result.setResultType(bool_result);

        // Note: This will break the display is successfull tests are
        // displayed
        if (!bool_result) {
            result.setLhs(Catch::toString(lhs));
            result.setRhs("Approx(" + Catch::toString(rhs) + ")");
            result.setOp("==");
        }

        result.endExpression();
        result.react();
#ifdef ETL_VERY_FAST_CATCH
    }
#endif
}

#define REQUIRE_DIRECT(value) \
    evaluate_result_direct(__FILE__, __LINE__, #value, value);

#define REQUIRE_EQUALS(lhs, rhs) \
    evaluate_result(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs);

#define REQUIRE_EQUALS_APPROX(lhs, rhs) \
    evaluate_result_approx(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs);

#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) \
    evaluate_result_approx(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs, eps);

#else

#define REQUIRE_DIRECT(value) REQUIRE(value)
#define REQUIRE_EQUALS(lhs, rhs) REQUIRE(lhs == rhs)
#define REQUIRE_EQUALS_APPROX(lhs, rhs) REQUIRE(lhs == Approx(rhs))
#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) REQUIRE(lhs == Approx(rhs).epsilon(eps))

#endif
