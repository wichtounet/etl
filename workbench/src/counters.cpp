//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#define ETL_COUNTERS

#include "etl/etl.hpp"

int main(){
    etl::dyn_matrix<float, 2> A(1000, 1000);
    etl::dyn_matrix<float, 2> B(1000, 1000);
    etl::dyn_matrix<float, 2> C(1000, 1000);
    etl::dyn_matrix<float, 2> D(1000, 1000);

    A = etl::normal_generator<float>(1.0, 0.0);
    B = etl::normal_generator<float>(1.0, 0.0);
    C = etl::normal_generator<float>(1.0, 0.0);
    D = etl::normal_generator<float>(1.0, 0.0);

    std::cout << "Simple" << std::endl;

    for(size_t i = 0; i < 10; ++i){
        C = A * B;
        std::cout << "sum:" << etl::sum(C) << std::endl;
    }

    etl::dump_counters();
    etl::reset_counters();

    std::cout << "Basic" << std::endl;

    for(size_t i = 0; i < 10; ++i){
        C = A * B;
        D += C;
        D *= 1.1;
        std::cout << "sum:" << etl::mean(D) << std::endl;
    }

    etl::dump_counters();
    etl::reset_counters();

    return 0;
}
