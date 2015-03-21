//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_EVALUATOR_HPP
#define ETL_EVALUATOR_HPP

#include "traits_fwd.hpp"   //forward declaration of the traits

namespace etl {

template<typename Expr, typename Result>
struct standard_evaluator {
    template<typename E = Expr, typename R = Result>
    static void evaluate(Expr&& expr, Result&& result){
        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] = expr[i];
        }
    }
};

template<typename Expr, typename Result>
void evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

} //end of namespace etl

#endif
