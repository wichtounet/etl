//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <array>     //To store the dimensions
#include <tuple>     //For TMP stuff
#include <algorithm> //For std::find_if
#include <iosfwd>    //For stream support

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/traits_lite.hpp" //forward declaration of the traits
#include "etl/compat.hpp"      //To make it work with g++

// CRTP classes
#include "etl/crtp/inplace_assignable.hpp"
#include "etl/crtp/comparable.hpp"
#include "etl/crtp/value_testable.hpp"
#include "etl/crtp/dim_testable.hpp"
#include "etl/crtp/expression_able.hpp"

namespace etl {

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

template <typename... V>
struct values_t {
    const std::tuple<V...> values;
    explicit values_t(V... v)
            : values(v...){};

    template <typename T, std::size_t... I>
    std::vector<T> list_sub(const std::index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }

    template <typename T>
    std::vector<T> list() const {
        return list_sub<T>(std::make_index_sequence<sizeof...(V)>());
    }
};

template <typename... V>
values_t<V...> values(V... v) {
    return values_t<V...>{v...};
}

namespace dyn_detail {

template <typename... S>
struct is_init_constructor : std::false_type {};

template <typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> : std::is_same<init_flag_t, typename cpp::nth_type<1 + sizeof...(S), S1, S2, S3, S...>::type> {};

template <typename... S>
struct is_initializer_list_constructor : std::false_type {};

template <typename S1, typename S2, typename... S>
struct is_initializer_list_constructor<S1, S2, S...> : cpp::is_specialization_of<std::initializer_list, typename cpp::last_type<S1, S2, S...>::type> {};

inline std::size_t size(std::size_t first) {
    return first;
}

template <typename... T>
inline std::size_t size(std::size_t first, T... args) {
    return first * size(args...);
}

template <std::size_t... I, typename... T>
inline std::size_t size(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return size((cpp::nth_value<I>(args...))...);
}

template <std::size_t... I, typename... T>
inline std::array<std::size_t, sizeof...(I)> sizes(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return {{static_cast<std::size_t>(cpp::nth_value<I>(args...))...}};
}

} // end of namespace dyn_detail

/*!
 * \brief Matrix with runn-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, order SO, std::size_t D>
struct dyn_matrix_impl final : inplace_assignable<dyn_matrix_impl<T, SO, D>>, comparable<dyn_matrix_impl<T, SO, D>>, expression_able<dyn_matrix_impl<T, SO, D>>, value_testable<dyn_matrix_impl<T, SO, D>>, dim_testable<dyn_matrix_impl<T, SO, D>> {
    static_assert(D > 0, "A matrix must have a least 1 dimension");

public:
    static constexpr const std::size_t n_dimensions = D;
    static constexpr const order storage_order      = SO;
    static constexpr const std::size_t alignment    = intrinsic_traits<T>::alignment;

    using value_type             = T;
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;
    using memory_type            = value_type*;
    using const_memory_type      = const value_type*;
    using iterator               = memory_type;
    using const_iterator         = const_memory_type;
    using vec_type               = intrinsic_type<T>;

private:
    std::size_t _size;
    dimension_storage_impl _dimensions;
    memory_type _memory;

    void check_invariants() {
        cpp_assert(_dimensions.size() == D, "Invalid dimensions");

#ifndef NDEBUG
        auto computed = std::accumulate(_dimensions.begin(), _dimensions.end(), std::size_t(1), std::multiplies<std::size_t>());
        cpp_assert(computed == _size, "Incoherency in dimensions");
#endif
    }

    static memory_type allocate(std::size_t n) {
        auto* memory = aligned_allocator<void, alignment>::template allocate<T>(n);
        cpp_assert(memory, "Impossible to allocate memory for dyn_matrix");
        cpp_assert(reinterpret_cast<uintptr_t>(memory) % alignment == 0, "Failed to align memory of matrix");

        //In case of non-trivial type, we need to call the constructors
        if(!std::is_trivial<value_type>::value){
            new (memory) value_type[n]();
        }

        return memory;
    }

    static void release(memory_type ptr, std::size_t n) {
        //In case of non-trivial type, we need to call the destructors
        if(!std::is_trivial<value_type>::value){
            for(std::size_t i = 0; i < n; ++i){
                ptr[i].~value_type();
            }
        }

        aligned_allocator<void, alignment>::template release<T>(ptr);
    }

public:
    // Construction

    //Default constructor (constructs an empty matrix)
    dyn_matrix_impl() noexcept : _size(0), _memory(nullptr) {
        std::fill(_dimensions.begin(), _dimensions.end(), 0);

        check_invariants();
    }

    //Copy constructor
    dyn_matrix_impl(const dyn_matrix_impl& rhs) noexcept : _size(rhs._size), _dimensions(rhs._dimensions), _memory(allocate(_size)) {
        check_invariants();

        std::copy_n(rhs._memory, _size, _memory);

    }

    //Copy constructor with different type
    //This constructor is necessary because the one from expression is explicit
    template <typename T2>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO, D>& rhs) noexcept : _size(rhs.size()), _memory(allocate(_size)) {
        //The type is different, therefore attributes are private
        for (std::size_t d = 0; d < etl::dimensions(rhs); ++d) {
            _dimensions[d] = etl::dim(rhs, d);
        }

        //The type is different, so we must use assign
        assign_evaluate(rhs, *this);

        check_invariants();
    }

    //Move constructor
    dyn_matrix_impl(dyn_matrix_impl&& rhs) noexcept : _size(rhs._size), _dimensions(std::move(rhs._dimensions)), _memory(rhs._memory) {
        rhs._size = 0;
        rhs._memory = nullptr;

        check_invariants();
    }

    //Initializer-list construction for vector
    dyn_matrix_impl(std::initializer_list<value_type> list) noexcept : _size(list.size()),
                                                                       _dimensions{{list.size()}},
                                                                       _memory(allocate(_size)) {
        static_assert(n_dimensions == 1, "This constructor can only be used for 1D matrix");

        std::copy(list.begin(), list.end(), begin());

        check_invariants();
    }

    //Normal constructor with only sizes
    template <typename... S, cpp_enable_if(
                                 (sizeof...(S) == D),
                                 cpp::all_convertible_to<std::size_t, S...>::value,
                                 cpp::is_homogeneous<typename cpp::first_type<S...>::type, S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : _size(dyn_detail::size(sizes...)),
                                                    _dimensions{{static_cast<std::size_t>(sizes)...}},
                                                                       _memory(allocate(_size))  {
        //Nothing to init

        check_invariants();
    }

    //Sizes followed by an initializer list
    template <typename... S, cpp_enable_if(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
                                                    _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
                                                                       _memory(allocate(_size))  {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        std::copy(list.begin(), list.end(), begin());

        check_invariants();
    }

    //Sizes followed by a values_t
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              cpp::is_specialization_of<values_t, typename cpp::last_type<S1, S...>::type>::value)>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                                       _memory(allocate(_size))  {
        auto list = cpp::last_value(sizes...).template list<value_type>();
        std::copy(list.begin(), list.end(), begin());

        check_invariants();
    }

    //Sizes followed by a value
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              std::is_convertible<std::size_t, typename cpp::first_type<S1, S...>::type>::value, //The first type must be convertible to size_t
                                              cpp::is_sub_homogeneous<S1, S...>::value,                                          //The first N-1 types must homegeneous
                                              (std::is_arithmetic<typename cpp::last_type<S1, S...>::type>::value
                                                   ? std::is_convertible<value_type, typename cpp::last_type<S1, S...>::type>::value //The last type must be convertible to value_type
                                                   : std::is_same<value_type, typename cpp::last_type<S1, S...>::type>::value        //The last type must be exactly value_type
                                               ))>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                                       _memory(allocate(_size))  {
        decltype(auto) value = cpp::last_value(s1, sizes...);
        std::fill(begin(), end(), value);
        check_invariants();
    }

    //Sizes followed by a generator_expr
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              std::is_convertible<std::size_t, typename cpp::first_type<S1, S...>::type>::value,        //The first type must be convertible to size_t
                                              cpp::is_sub_homogeneous<S1, S...>::value,                                                 //The first N-1 types must homegeneous
                                              cpp::is_specialization_of<generator_expr, typename cpp::last_type<S1, S...>::type>::value //The last type must be a generator expr
                                              )>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                                       _memory(allocate(_size))  {
        const auto& e = cpp::last_value(sizes...);

        assign_evaluate(e, *this);

        check_invariants();
    }

    //Sizes followed by an init flag followed by the value
    template <typename... S, cpp_enable_if(dyn_detail::is_init_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : _size(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
                                                    _dimensions(dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
                                                                       _memory(allocate(_size))  {
        static_assert(sizeof...(S) == D + 2, "Invalid number of dimensions");

        std::fill(begin(), end(), cpp::last_value(sizes...));

        check_invariants();
    }

    template <typename E, cpp_enable_if(
                              std::is_convertible<value_t<E>, value_type>::value,
                              is_etl_expr<E>::value)>
    explicit dyn_matrix_impl(E&& e)
            : _size(etl::size(e)), _memory(allocate(_size)) {
        for (std::size_t d = 0; d < etl::dimensions(e); ++d) {
            _dimensions[d] = etl::dim(e, d);
        }

        assign_evaluate(std::forward<E>(e), *this);

        check_invariants();
    }

    template <typename Container, cpp_enable_if(
                                      cpp::not_c<is_etl_expr<Container>>::value,
                                      std::is_convertible<typename Container::value_type, value_type>::value)>
    explicit dyn_matrix_impl(const Container& vec)
            : _size(vec.size()), _dimensions{{_size}, _memory(allocate(_size))} {
        static_assert(D == 1, "Only 1D matrix can be constructed from containers");

        for (std::size_t i = 0; i < size(); ++i) {
            _memory[i] = vec[i];
        }

        check_invariants();
    }

    // Assignment

    //Copy assignment operator

    //Note: For now, this is the only constructor that is able to change the size and dimensions of the matrix
    dyn_matrix_impl& operator=(const dyn_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            if (!_size) {
                _size       = rhs.size();
                _dimensions = rhs._dimensions;
                _memory =   allocate(_size);
                std::copy_n(rhs._memory, _size, _memory);
            } else {
                validate_assign(*this, rhs);
                assign_evaluate(rhs, *this);
            }
        }

        check_invariants();

        return *this;
    }

    //Default move assignment operator
    dyn_matrix_impl& operator=(dyn_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            if(_memory){
                release(_memory, _size);
            }

            _size       = rhs._size;
            _dimensions = std::move(rhs._dimensions);
            _memory = rhs._memory;

            rhs._size = 0;
            rhs._memory = nullptr;
        }

        check_invariants();

        return *this;
    }

    //Construct from expression

    template <typename E, cpp_enable_if(!std::is_same<std::decay_t<E>, dyn_matrix_impl<T, SO, D>>::value && std::is_convertible<value_t<E>, value_type>::value && is_etl_expr<E>::value)>
    dyn_matrix_impl& operator=(E&& e) {
        validate_assign(*this, e);

        assign_evaluate(e, *this);

        check_invariants();

        return *this;
    }

    //Allow copy from other containers

    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    dyn_matrix_impl& operator=(const Container& vec) {
        validate_assign(*this, vec);

        std::copy(vec.begin(), vec.end(), begin());

        check_invariants();

        return *this;
    }

    //Set the same value to each element of the matrix
    dyn_matrix_impl& operator=(const value_type& value) {
        std::fill(begin(), end(), value);

        check_invariants();

        return *this;
    }

    //Destructor

    ~dyn_matrix_impl(){
        if(_memory){
            release(_memory, _size);
        }
    }

    // Swap operations

    void swap(dyn_matrix_impl& other) {
        using std::swap;
        swap(_size, other._size);
        swap(_dimensions, other._dimensions);
        swap(_memory, other._memory);

        check_invariants();
    }

    // Accessors

    std::size_t size() const noexcept {
        return _size;
    }

    std::size_t rows() const noexcept {
        return _dimensions[0];
    }

    std::size_t columns() const noexcept {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }

    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
    }

    std::size_t dim(std::size_t d) const noexcept {
        cpp_assert(d < n_dimensions, "Invalid dimension");

        return _dimensions[d];
    }

    template <std::size_t D2>
    std::size_t dim() const noexcept {
        cpp_assert(D2 < n_dimensions, "Invalid dimension");

        return _dimensions[D2];
    }

    template <bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) noexcept {
        return sub(*this, i);
    }

    template <bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const noexcept {
        return sub(*this, i);
    }

    template <bool B = n_dimensions == 1, cpp::enable_if_u<B> = cpp::detail::dummy>
    value_type& operator()(std::size_t i) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

    template <bool B = n_dimensions == 1, cpp::enable_if_u<B> = cpp::detail::dummy>
    const value_type& operator()(std::size_t i) const noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

    template <bool B = n_dimensions == 2, cpp::enable_if_u<B> = cpp::detail::dummy>
    value_type& operator()(std::size_t i, std::size_t j) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        if (storage_order == order::RowMajor) {
            return _memory[i * dim(1) + j];
        } else {
            return _memory[j * dim(0) + i];
        }
    }

    template <bool B = n_dimensions == 2, cpp::enable_if_u<B> = cpp::detail::dummy>
    const value_type& operator()(std::size_t i, std::size_t j) const noexcept {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        if (storage_order == order::RowMajor) {
            return _memory[i * dim(1) + j];
        } else {
            return _memory[j * dim(0) + i];
        }
    }

    template <typename... S, cpp_enable_if((sizeof...(S) > 0))>
    std::size_t index(S... sizes) const noexcept {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        std::size_t index = 0;

        if (storage_order == order::RowMajor) {
            std::size_t subsize = size();
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    subsize /= dim(i++);
                    index += subsize * s;
                },
                sizes...);
        } else {
            std::size_t subsize = 1;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    index += subsize * s;
                    subsize *= dim(i++);
                },
                sizes...);
        }

        return index;
    }

    template <typename... S, cpp::enable_if_all_u<
                                 (n_dimensions > 2),
                                 (sizeof...(S) == n_dimensions),
                                 cpp::all_convertible_to<std::size_t, S...>::value> = cpp::detail::dummy>
    const value_type& operator()(S... sizes) const noexcept {
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _memory[index(sizes...)];
    }

    template <typename... S, cpp::enable_if_all_u<
                                 (n_dimensions > 2),
                                 (sizeof...(S) == n_dimensions),
                                 cpp::all_convertible_to<std::size_t, S...>::value> = cpp::detail::dummy>
    value_type& operator()(S... sizes) noexcept {
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _memory[index(sizes...)];
    }

    const value_type& operator[](std::size_t i) const noexcept {
        cpp_assert(i < size(), "Out of bounds");

        return _memory[i];
    }

    value_type& operator[](std::size_t i) noexcept {
        cpp_assert(i < size(), "Out of bounds");

        return _memory[i];
    }

    vec_type load(std::size_t i) const noexcept {
        return vec::loadu(_memory + i);
    }

    iterator begin() noexcept {
        return _memory;
    }

    iterator end() noexcept {
        return _memory + _size;
    }

    const_iterator begin() const noexcept {
        return _memory;
    }

    const_iterator end() const noexcept {
        return _memory + _size;
    }

    const_iterator cbegin() const noexcept {
        return _memory;
    }

    const_iterator cend() const noexcept {
        return _memory + _size;
    }

    // Direct memory access

    inline memory_type memory_start() noexcept {
        return _memory;
    }

    inline const_memory_type memory_start() const noexcept {
        return _memory;
    }

    memory_type memory_end() noexcept {
        return _memory + _size;
    }

    const_memory_type memory_end() const noexcept {
        return _memory + _size;
    }

    std::size_t& unsafe_dimension_access(std::size_t i) {
        cpp_assert(i < n_dimensions, "Out of bounds");
        return _dimensions[i];
    }
};

template <typename T, order SO, std::size_t D>
void swap(dyn_matrix_impl<T, SO, D>& lhs, dyn_matrix_impl<T, SO, D>& rhs) {
    lhs.swap(rhs);
}

template <typename T, order SO, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_impl<T, SO, D>& mat) {
    if (D == 1) {
        return os << "V[" << mat.size() << "]";
    }

    os << "M[" << mat.dim(0);

    for (std::size_t i = 1; i < D; ++i) {
        os << "," << mat.dim(i);
    }

    return os << "]";
}

} //end of namespace etl
