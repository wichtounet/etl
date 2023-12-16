//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "etl/etl.hpp"

etl::fast_vector<double, 11> a = {-1.0, 2.0, 5.0, 2.0, 5.0, 1.2, 2.5, 1.2, -3.0, 3.5, 1.0};
etl::fast_vector<double, 11> b = {-1.0, 2.0, 5.0, 1.2, 2.5, 3.0, 4.0, 1.2, -3.0, 3.5, 1.0};
etl::fast_vector<double, 11> c = {1.2, -3.0, 3.5, 1.2, -3.0, 3.5, 1.0};

/*
 * Simple source file to verify how the code is compiled.
 */

int main(){
    etl::fast_vector<double, 3> d((1.5 * (a * b + c)) / a);

    return static_cast<int>(sum(d));
}
