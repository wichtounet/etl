//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

inline void evaluate_result_direct(Catch::ResultBuilder&& __result, bool value){
    __result.setResultType(value);
    __result.setLhs(value ? "true" : "false");
    __result.setOp("");
    __result.endExpression();
    __result.react();
}

template<typename L, typename R>
void evaluate_result(Catch::ResultBuilder&& __result, L lhs, R rhs){
    __result.setResultType(lhs == rhs);
    __result.setLhs(Catch::toString(lhs));
    __result.setRhs(Catch::toString(rhs));
    __result.setOp("==");
    __result.endExpression();
    __result.react();
}

template<typename L, typename R>
void evaluate_result_approx(Catch::ResultBuilder&& __result, L lhs, R rhs, double eps = std::numeric_limits<float>::epsilon()*100){
    __result.setResultType(fabs(lhs - rhs) < eps * (1.0 + std::max(fabs(lhs), fabs(rhs))));
    __result.setLhs(Catch::toString(lhs));
    __result.setRhs("Approx(" + Catch::toString(rhs) + ")");
    __result.setOp("==");
    __result.endExpression();
    __result.react();
}

#define REQUIRE_DIRECT(value) \
    evaluate_result_direct(Catch::ResultBuilder( "REQUIRE", CATCH_INTERNAL_LINEINFO, #value, Catch::ResultDisposition::Normal ), value);

#define REQUIRE_EQUALS(lhs, rhs) \
    evaluate_result(Catch::ResultBuilder( "REQUIRE", CATCH_INTERNAL_LINEINFO, #lhs " == " #rhs, Catch::ResultDisposition::Normal ), lhs, rhs);

#define REQUIRE_EQUALS_APPROX(lhs, rhs) \
    evaluate_result_approx(Catch::ResultBuilder( "REQUIRE", CATCH_INTERNAL_LINEINFO, #lhs " == " #rhs, Catch::ResultDisposition::Normal ), lhs, rhs);

#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) \
    evaluate_result_approx(Catch::ResultBuilder( "REQUIRE", CATCH_INTERNAL_LINEINFO, #lhs " == " #rhs, Catch::ResultDisposition::Normal ), lhs, rhs, eps);
