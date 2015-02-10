//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DYN_MATRIX_HPP
#define ETL_DYN_MATRIX_HPP

#include <vector> //To store the data
#include <array>  //To store the dimensions
#include <tuple>  //For TMP stuff

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "traits_fwd.hpp"

namespace etl {

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

template<typename... V>
struct values_t {
    const std::tuple<V...> values;
    values_t(V... v) : values(v...) {};

    template<typename T, std::size_t... I>
    std::vector<T> list_sub(const std::index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }

    template<typename T>
    std::vector<T> list() const {
        return list_sub<T>(std::make_index_sequence<sizeof...(V)>());
    }
};

template<typename... V>
values_t<V...> values(V... v){
    return {v...};
}

namespace dyn_detail {

template<typename... S>
struct is_init_constructor : std::integral_constant<bool, false> {};

template<typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> :
    std::integral_constant<bool, std::is_same<init_flag_t, typename cpp::nth_type<1+sizeof...(S), S1, S2, S3, S...>::type>::value> {};

template<typename... S>
struct is_initializer_list_constructor : std::integral_constant<bool, false> {};

template<typename S1, typename S2, typename... S>
struct is_initializer_list_constructor<S1, S2, S...> :
    std::integral_constant<bool, cpp::is_specialization_of<std::initializer_list, typename cpp::last_type<S1, S2, S...>::type>::value> {};

inline std::size_t size(std::size_t first){
    return first;
}

template<typename... T>
inline std::size_t size(std::size_t first, T... args){
    return first * size(args...);
}

template<std::size_t... I, typename... T>
inline std::size_t size(const std::index_sequence<I...>& /*i*/, const T&... args){
    return size((cpp::nth_value<I>(args...))...);
}

template<std::size_t... I, typename... T>
inline std::array<std::size_t, sizeof...(I)> sizes(const std::index_sequence<I...>& /*i*/, const T&... args){
    return {{static_cast<std::size_t>(cpp::nth_value<I>(args...))...}};
}

} // end of namespace dyn_detail

template<typename T, std::size_t D = 2>
struct dyn_matrix final {
    static_assert(D > 0, "A matrix must have a least 1 dimension");

public:
    static constexpr const std::size_t n_dimensions = D;

    using                value_type = T;
    using              storage_impl = std::vector<value_type>;
    using    dimension_storage_impl = std::array<std::size_t, n_dimensions>;
    using                  iterator = typename storage_impl::iterator;
    using            const_iterator = typename storage_impl::const_iterator;

private:
    const std::size_t _size;
    storage_impl _data;
    dimension_storage_impl _dimensions;

public:
    //{{{ Construction

    dyn_matrix(const dyn_matrix& rhs) : _size(rhs._size), _data(rhs._data), _dimensions(rhs._dimensions) {
        //Nothing to init
    }

    //Sizes followed by an initializer list
    dyn_matrix(std::initializer_list<value_type> list) :
            _size(list.size()),
            _data(list),
            _dimensions{{list.size()}} {
        static_assert(n_dimensions == 1, "This constructor can only be used for 1D matrix");
        //Nothing to init
    }

    //Normal constructor with only sizes
    template<typename... S, cpp::enable_if_all_u<
            (sizeof...(S) == D),
            cpp::all_convertible_to<std::size_t, S...>::value,
            cpp::is_homogeneous<typename cpp::first_type<S...>::type, S...>::value
        > = cpp::detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(sizes...)),
            _data(_size),
            _dimensions{{static_cast<std::size_t>(sizes)...}} {
        //Nothing to init
    }

    //Sizes followed by an initializer list
    template<typename... S, cpp::enable_if_u<dyn_detail::is_initializer_list_constructor<S...>::value> = cpp::detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
            _data(cpp::last_value(sizes...)),
            _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");
    }

    //Sizes followed by a values_t
    template<typename S1, typename... S, cpp::enable_if_all_u<
            (sizeof...(S) == D),
            cpp::is_specialization_of<values_t, typename cpp::last_type<S1, S...>::type>::value
        > = cpp::detail::dummy>
    dyn_matrix(S1 s1, S... sizes) :
            _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
            _data(cpp::last_value(s1, sizes...).template list<value_type>()),
            _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        //Nothing to init
    }

    //Sizes followed by a value
    template<typename S1, typename... S, cpp::enable_if_all_u<
            (sizeof...(S) == D),
            std::is_convertible<std::size_t, typename cpp::first_type<S1, S...>::type>::value,   //The first type must be convertible to size_t
            cpp::is_sub_homogeneous<S1, S...>::value,                                            //The first N-1 types must homegeneous
            std::is_same<value_type, typename cpp::last_type<S1, S...>::type>::value             //The last type must be exactly value_type
        > = cpp::detail::dummy>
    dyn_matrix(S1 s1, S... sizes) :
            _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
            _data(_size, cpp::last_value(s1, sizes...)),
            _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        //Nothing to init
    }

    //Sizes followed by a generator_expr
    template<typename S1, typename... S, cpp::enable_if_all_u<
            (sizeof...(S) == D),
            std::is_convertible<std::size_t, typename cpp::first_type<S1, S...>::type>::value,   //The first type must be convertible to size_t
            cpp::is_sub_homogeneous<S1, S...>::value,                                            //The first N-1 types must homegeneous
            cpp::is_specialization_of<generator_expr, typename cpp::last_type<S1, S...>::type>::value     //The last type must be a generator expr
        > = cpp::detail::dummy>
    dyn_matrix(S1 s1, S... sizes) :
            _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
            _data(_size),
            _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        const auto& e = cpp::last_value(sizes...);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    //Sizes followed by an init flag followed by the value
    template<typename... S, cpp::enable_if_c<dyn_detail::is_init_constructor<S...>> = cpp::detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
            _data(_size, cpp::last_value(sizes...)),
            _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)) {
        static_assert(sizeof...(S) == D + 2, "Invalid number of dimensions");

        //Nothing to init
    }

    template<typename E, cpp::enable_if_all_c<
        std::is_convertible<typename E::value_type, value_type>,
        is_copy_expr<E>
    > = cpp::detail::dummy>
    dyn_matrix(const E& e) :_size(etl::size(e)), _data(_size) {
        for(std::size_t d = 0; d < etl::dimensions(e); ++d){
            _dimensions[d] = etl::dim(e, d);
        }

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename Container, cpp::enable_if_all_c<
        cpp::not_c<is_etl_expr<Container>>,
        std::is_convertible<typename Container::value_type, value_type>
    > = cpp::detail::dummy>
    dyn_matrix(const Container& vec) : _size(vec.size()), _data(_size), _dimensions{{_size}} {
        static_assert(D == 1, "Only 1D matrix can be constructed from containers");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = vec[i];
        }
    }

    //Default move
    dyn_matrix(dyn_matrix&& rhs) = default;

    //}}}

    //{{{ Assignment

    //Copy assignment operator

    dyn_matrix& operator=(const dyn_matrix& rhs){
        ensure_same_size(*this, rhs);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Construct from expression

    template<typename E, cpp::enable_if_all_c<
        std::is_convertible<typename E::value_type, value_type>,
        is_copy_expr<E>
    > = cpp::detail::dummy>
    dyn_matrix& operator=(E&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename Generator>
    dyn_matrix& operator=(generator_expr<Generator>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, cpp::enable_if_all_c<
        cpp::not_c<is_etl_expr<Container>>,
        std::is_convertible<typename Container::value_type, value_type>
    > = cpp::detail::dummy>
    dyn_matrix& operator=(const Container& vec){
        cpp_assert(vec.size() == size(), "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //Set the same value to each element of the matrix
    dyn_matrix& operator=(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Default move
    dyn_matrix& operator=(dyn_matrix&& rhs) = default;

    //}}}

    void swap(dyn_matrix& other){
        cpp_assert(other.size() == size(), "Cannot swap from a dyn_matrix of different size");

        std::swap(_data, other._data);
        std::swap(_dimensions, other._dimensions);
    }

    //{{{ Accessors

    std::size_t size() const noexcept {
        return _size;
    }

    std::size_t rows() const {
        return _dimensions[0];
    }

    std::size_t columns() const {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }

    static constexpr std::size_t dimensions(){
        return n_dimensions;
    }

    std::size_t dim(std::size_t d) const {
        cpp_assert(d < n_dimensions, "Invalid dimension");

        return _dimensions[d];
    }

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    template<bool B = n_dimensions == 1, cpp::enable_if_u<B> = cpp::detail::dummy>
    value_type& operator()(std::size_t i){
        cpp_assert(i < dim(0), "Out of bounds");

        return _data[i];
    }

    template<bool B = n_dimensions == 1, cpp::enable_if_u<B> = cpp::detail::dummy>
    const value_type& operator()(std::size_t i) const {
        cpp_assert(i < dim(0), "Out of bounds");

        return _data[i];
    }

    template<bool B = n_dimensions == 2, cpp::enable_if_u<B> = cpp::detail::dummy>
    value_type& operator()(std::size_t i, std::size_t j){
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        return _data[i * dim(1) + j];
    }

    template<bool B = n_dimensions == 2, cpp::enable_if_u<B> = cpp::detail::dummy>
    const value_type& operator()(std::size_t i, std::size_t j) const {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        return _data[i * dim(1) + j];
    }

    template<typename... S, cpp::enable_if_u<(sizeof...(S) > 0)> = cpp::detail::dummy>
    std::size_t index(S... sizes) const {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        auto subsize = size();
        std::size_t index = 0;
        std::size_t i = 0;

        cpp::for_each_in(
            [&subsize, &index, &i, this](std::size_t s){
                subsize /= dim(i++);
                index += subsize * s;
            }, sizes...);

        return index;
    }

    template<typename... S, cpp::enable_if_all_u<
            (n_dimensions > 2),
            (sizeof...(S) == n_dimensions),
            cpp::all_convertible_to<std::size_t, S...>::value
        > = cpp::detail::dummy>
    const value_type& operator()(S... sizes) const {
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _data[index(sizes...)];
    }

    template<typename... S, cpp::enable_if_all_u<
            (n_dimensions > 2),
            (sizeof...(S) == n_dimensions),
            cpp::all_convertible_to<std::size_t, S...>::value
        > = cpp::detail::dummy>
    value_type& operator()(S... sizes){
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _data[index(sizes...)];
    }

    const value_type& operator[](std::size_t i) const {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    value_type& operator[](std::size_t i){
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    const_iterator begin() const {
        return _data.begin();
    }

    const_iterator end() const {
        return _data.end();
    }

    iterator begin(){
        return _data.begin();
    }

    iterator end(){
        return _data.end();
    }

    //}}}
};

template<typename T, std::size_t D>
void swap(dyn_matrix<T, D>& lhs, dyn_matrix<T, D>& rhs){
    lhs.swap(rhs);
}

} //end of namespace etl

#endif
