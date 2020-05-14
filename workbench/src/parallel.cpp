//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::microseconds microseconds;

namespace {

template<typename T>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(auto& v : container){
        v = generator();
    }
}

template<typename T1>
void randomize(T1& container){
    randomize_double(container);
}

template<typename T1, typename... TT>
void randomize(T1& container, TT&... containers){
    randomize_double(container);
    randomize(containers...);
}

std::string duration_str(size_t duration_us){
    double duration = duration_us;

    if(duration > 1000 * 1000){
        return std::to_string(duration / 1000.0 / 1000.0) + "s";
    } else if(duration > 1000){
        return std::to_string(duration / 1000.0) + "ms";
    } else {
        return std::to_string(duration_us) + "us";
    }
}

template<typename Functor, typename... T>
void measure(const std::string& title, const std::string& reference, Functor&& functor, T&... references){
    for(size_t i = 0; i < 100; ++i){
        randomize(references...);
        functor();
    }

    size_t duration_acc = 0;

    for(size_t i = 0; i < 100; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")\n";
}

void bench(size_t n){
    etl::dyn_matrix<double, 1> a(n), b(n), c(n);

    randomize(a, b, c);

    size_t duration_acc = 0;

    auto steps = 100UL;
    auto rpt = 1000UL;

    for(size_t i = 0; i < steps; ++i){
        auto start_time = timer_clock::now();

        for(size_t i = 0; i < rpt; ++i){
            c = a + b;
        }

        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }


    std::cout << "Size:" << n << " took " << duration_str(duration_acc) << "\n";
    std::cout << "\tMFlops:" << n  / (double(duration_acc)  / rpt / double(steps)) << std::endl;
}

} //end of anonymous namespace

int main(){
    for(size_t n = 90000; n <= 110000; n += 10000){
        bench( n );
    }

    return 0;
}
