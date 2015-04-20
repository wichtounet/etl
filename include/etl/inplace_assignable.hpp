//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_INPLACE_ASSIGNABLE_HPP
#define ETL_INPLACE_ASSIGNABLE_HPP

/*
 * Use CRTP technique to inject inplace operations into expressions and value classes. 
 */

namespace etl {

template<typename D>
struct inplace_assignable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    template<typename E>
    derived_t& scale_inplace(E&& e){
        as_derived() *= e;

        return as_derived();
    }

    derived_t& fflip_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() <= 2, "Impossible to fflip a matrix of D > 2");

        if(etl_traits<derived_t>::dimensions() == 2){
            std::reverse(as_derived().begin(), as_derived().end());
        }

        return as_derived();
    }

    template<typename S = D, cpp_disable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");
        cpp_assert(etl::dim<0>(as_derived()) == etl::dim<1>(as_derived()), "Only square matrices can be tranposed inplace");

        const auto N = etl::dim<0>(as_derived());

        for(std::size_t i = 0; i < N - 1; ++i){
            for(std::size_t j = i + 1; j < N; ++j){
                using std::swap;
                swap(as_derived()(i, j), as_derived()(j, i));
            }
        }

        return as_derived();
    }

    template<typename S = D, cpp_enable_if(is_dyn_matrix<S>::value)>
    derived_t& transpose_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be transposed");

        using std::swap;

        decltype(auto) mat = as_derived();

        const auto N = etl::dim<0>(mat);
        const auto M = etl::dim<1>(mat);

        if(M == N){
            for(std::size_t i = 0; i < N - 1; ++i){
                for(std::size_t j = i + 1; j < N; ++j){
                    swap(mat(i, j), mat(j, i));
                }
            }
        } else {
            swap(mat.unsafe_dimension_access(0), mat.unsafe_dimension_access(1));

            auto data = mat.memory_start();

            for(std::size_t k = 0; k < N*M; k++) {
                auto idx = k;
                do {
                    idx = (idx % N) * M + (idx / N);
                } while(idx < k);
                std::swap(data[k], data[idx]);
            }
        }

        return mat;
    }
};

} //end of namespace etl

#endif
