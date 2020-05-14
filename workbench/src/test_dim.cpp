//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "etl/etl.hpp"

etl::fast_matrix<double, 2, 3> a = {-1.0, 2.0, 5.0, 2.0, 5.0, 1.2};
etl::fast_matrix<double, 2, 3> b = {-1.0, 2.0, 5.0, 1.2, 2.5, 3.0};

/*
 * Simple source file to verify how the code is compiled.
 */

int main(){
    etl::fast_vector<double, 3> d(row(a,1) + 1.5 * (row(a,0) / 2.0 + row(b,1)));

    return static_cast<int>(sum(d));
}
