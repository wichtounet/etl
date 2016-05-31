//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

constexpr const auto base_eps = std::numeric_limits<float>::epsilon()*100;

inline void evaluate_result_direct(const char* file, std::size_t line, const char* exp, bool value){
    Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
    result.setResultType(value);
    result.setLhs(value ? "true" : "false");
    result.setOp("");
    result.endExpression();
    result.react();
}

template<typename L, typename R>
void evaluate_result(const char* file, std::size_t line, const char* exp, L lhs, R rhs){
    Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
    result.setResultType(lhs == rhs);
    result.setLhs(Catch::toString(lhs));
    result.setRhs(Catch::toString(rhs));
    result.setOp("==");
    result.endExpression();
    result.react();
}

template<typename L, typename R>
void evaluate_result_approx(const char* file, std::size_t line, const char* exp, L lhs, R rhs, double eps = base_eps){
    Catch::ResultBuilder result("REQUIRE", {file, line}, exp, Catch::ResultDisposition::Flags::Normal);
    result.setResultType(fabs(lhs - rhs) < eps * (1.0 + std::max(fabs(lhs), fabs(rhs))));
    result.setLhs(Catch::toString(lhs));
    result.setRhs("Approx(" + Catch::toString(rhs) + ")");
    result.setOp("==");
    result.endExpression();
    result.react();
}

#define REQUIRE_DIRECT(value) \
    evaluate_result_direct(__FILE__, __LINE__, #value, value);

#define REQUIRE_EQUALS(lhs, rhs) \
    evaluate_result(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs);

#define REQUIRE_EQUALS_APPROX(lhs, rhs) \
    evaluate_result_approx(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs);

#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) \
    evaluate_result_approx(__FILE__, __LINE__, #lhs " == " #rhs, lhs, rhs, eps);
