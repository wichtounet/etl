//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Base class and utilities for dyn matrix implementations
 */

#pragma once

#include "etl/index.hpp"

namespace etl {

/*!
 * \brief A simple type to use as init flag to constructor
 */
enum class init_flag_t {
    DUMMY  ///< Dummy value for the flag
};

/*!
 * \brief A simple value to use as init flag to constructor
 */
constexpr init_flag_t init_flag = init_flag_t::DUMMY;

/*!
 * \brief Simple collection of values to initialize a dyn matrix
 */
template <typename... V>
struct values_t {
    const std::tuple<V...> values; ///< The contained values

    /*!
     * \brief Construct a new sequence of values.
     */
    explicit values_t(V... v)
            : values(v...){};

    /*!
     * \brief Returns the sequence of values as a std::vector
     * \return a std::vector containing all the values
     */
    template <typename T>
    std::vector<T> list() const {
        return list_sub<T>(std::make_index_sequence<sizeof...(V)>());
    }

private:
    /*!
     * \brief Returns the sequence of values as a std::vector
     */
    template <typename T, size_t... I>
    std::vector<T> list_sub(const std::index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }
};

/*!
 * \brief Create a list of values for initializing a dyn_matrix
 */
template <typename... V>
values_t<V...> values(V... v) {
    return values_t<V...>{v...};
}

namespace dyn_detail {

/*!
 * \brief Traits to test if the constructor is an init constructor
 */
template <typename... S>
struct is_init_constructor : std::false_type {};

/*!
 * \copydoc is_init_list_constructor
 */
template <typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> : std::is_same<init_flag_t, typename cpp::nth_type<sizeof...(S), S2, S3, S...>::type> {};

/*!
 * \brief Traits to test if the constructor is an initializer list constructor
 */
template <typename... S>
struct is_initializer_list_constructor : std::false_type {};

/*!
 * \copydoc is_initializer_list_constructor
 */
template <typename S1, typename S2, typename... S>
struct is_initializer_list_constructor<S1, S2, S...> : cpp::is_specialization_of<std::initializer_list, typename cpp::last_type<S2, S...>::type> {};

/*!
 * \brief Returns a collection of dimensions of the matrix.
 */
template <size_t... I, typename... T>
inline std::array<size_t, sizeof...(I)> sizes(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return {{static_cast<size_t>(cpp::nth_value<I>(args...))...}};
}

} // end of namespace dyn_detail

/*!
 * \brief Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename Derived, typename T, size_t D>
struct dyn_base {
    static_assert(D > 0, "A matrix must have a least 1 dimension");

protected:
    static constexpr size_t n_dimensions = D;                                      ///< The number of dimensions
    static constexpr size_t alignment    = default_intrinsic_traits<T>::alignment; ///< The memory alignment

    using value_type             = T;                                ///< The value type
    using dimension_storage_impl = std::array<size_t, n_dimensions>; ///< The type used to store the dimensions
    using memory_type            = value_type*;                      ///< The memory type
    using const_memory_type      = const value_type*;                ///< The const memory type
    using derived_t              = Derived;                          ///< The derived (CRTP) type
    using this_type              = dyn_base<Derived, T, D>;          ///< The type of this class

    size_t _size;                  ///< The size of the matrix
    dimension_storage_impl _dimensions; ///< The dimensions of the matrix

    /*!
     * \brief Verify some invariants with assertions
     *
     * This function should only be used internally to ensure that
     * no breaking changes are made.
     */
    void check_invariants() {
        cpp_assert(_dimensions.size() == D, "Invalid dimensions");

#ifndef NDEBUG
        auto computed = std::accumulate(_dimensions.begin(), _dimensions.end(), size_t(1), std::multiplies<size_t>());
        cpp_assert(computed == _size, "Incoherency in dimensions");
#endif
    }

    /*!
     * \brief Allocate aligned memory for n elements of the given type
     * \tparam M the type of objects to allocate
     * \param n The number of elements to allocate
     * \return The allocated memory
     */
    template <typename M = value_type>
    static M* allocate(size_t n) {
        inc_counter("cpu:allocate");

        M* memory = aligned_allocator<alignment>::template allocate<M>(n);

        cpp_assert(memory, "Impossible to allocate memory for dyn_matrix");
        cpp_assert(reinterpret_cast<uintptr_t>(memory) % alignment == 0, "Failed to align memory of matrix");

        //In case of non-trivial type, we need to call the constructors
        if constexpr (!std::is_trivial<M>::value) {
            new (memory) M[n]();
        }

        if constexpr (padding){
            std::fill_n(memory, n, M());
        }

        return memory;
    }

    /*!
     * \brief Release aligned memory for n elements of the given type
     * \param ptr Pointer to the memory to release
     * \param n The number of elements to release
     */
    template <typename M>
    static void release(M* ptr, size_t n) {
        //In case of non-trivial type, we need to call the destructors
        if constexpr (!std::is_trivial<M>::value) {
            for (size_t i = 0; i < n; ++i) {
                ptr[i].~M();
            }
        }

        aligned_allocator<alignment>::template release<M>(ptr);
    }

    /*!
     * \brief Initialize the dyn_base with a size of 0
     */
    dyn_base() noexcept : _size(0) {
        std::fill(_dimensions.begin(), _dimensions.end(), 0);

        check_invariants();
    }

    /*!
     * \brief Copy construct a dyn_base
     * \param rhs The dyn_base to copy from
     */
    dyn_base(const dyn_base& rhs) noexcept = default;

    /*!
     * \brief Move construct a dyn_base
     * \param rhs The dyn_base to move from
     */
    dyn_base(dyn_base&& rhs) noexcept = default;

    /*!
     * \brief Construct a dyn_base if the given size and dimensions
     * \param size The size of the matrix
     * \param dimensions The dimensions of the matrix
     */
    dyn_base(size_t size, dimension_storage_impl dimensions) noexcept : _size(size), _dimensions(dimensions) {
        check_invariants();
    }

    /*!
     * \brief Move construct a dyn_base
     * \param rhs The dyn_base to move from
     */
    template <typename E, cpp_enable_iff(!std::is_same<std::decay_t<E>, derived_t>::value)>
    explicit dyn_base(E&& rhs) : _size(etl::size(rhs)) {
        for (size_t d = 0; d < etl::dimensions(rhs); ++d) {
            _dimensions[d] = etl::dim(rhs, d);
        }

        check_invariants();
    }

public:
    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr size_t dimensions() noexcept {
        return n_dimensions;
    }

    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    size_t size() const noexcept {
        return _size;
    }

    /*!
     * \brief Returns the number of rows of the matrix (the first dimension)
     * \return The number of rows of the matrix
     */
    size_t rows() const noexcept {
        return _dimensions[0];
    }

    /*!
     * \brief Returns the number of columns of the matrix (the first dimension)
     * \return The number of columns of the matrix
     */
    size_t columns() const noexcept {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }

    /*!
     * \brief Returns the dth dimension of the matrix
     * \param d The dimension to get
     * \return The Dth dimension of the matrix
     */
    size_t dim(size_t d) const noexcept(assert_nothrow) {
        cpp_assert(d < n_dimensions, "Invalid dimension");

        return _dimensions[d];
    }

    /*!
     * \brief Returns the D2th dimension of the matrix
     * \return The D2th dimension of the matrix
     */
    template <size_t D2>
    size_t dim() const noexcept(assert_nothrow) {
        cpp_assert(D2 < n_dimensions, "Invalid dimension");

        return _dimensions[D2];
    }
};

/*!
 *
 * \brief Dense Matrix with run-time fixed dimensions.
 * The matrix support an arbitrary number of dimensions.
 */
template <typename Derived, typename T, order SO, size_t D>
struct dense_dyn_base : dyn_base<Derived, T, D> {
    using value_type        = T;                                 ///< The type of the contained values
    using base_type         = dyn_base<Derived, T, D>;                    ///< The base type
    using this_type         = dense_dyn_base<Derived, T, SO, D>; ///< The type of this class
    using derived_t         = Derived;                           ///< The derived type
    using memory_type       = value_type*;                       ///< The memory type
    using const_memory_type = const value_type*;                 ///< The const memory type
    using iterator          = memory_type;                       ///< The type of iterator
    using const_iterator    = const_memory_type;                 ///< The type of const iterator

    using dimension_storage_impl = typename base_type::dimension_storage_impl; ///< The storage type used to store the dimensions

    static constexpr size_t n_dimensions = D;  ///< The number of dimensions
    static constexpr order storage_order      = SO; ///< The storage order

    using base_type::_size;
    using base_type::dim;

    value_type* ETL_RESTRICT _memory = nullptr; ///< Pointer to the allocated memory
    gpu_memory_handler<T> _gpu;                 ///< The GPU memory handler

    /*!
     * \brief Initialize the dense_dyn_base with a size of 0
     */
    dense_dyn_base() noexcept : base_type() {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a dense_dyn_base
     * \param rhs The dense_dyn_base to copy from
     */
    dense_dyn_base(const dense_dyn_base& rhs) noexcept : base_type(rhs), _gpu(rhs._gpu) {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a derived_t
     *
     * This constructor is necessary in order to use the correct constructor in
     * the parent type.
     *
     * \param rhs The derived_t to copy from
     */
    explicit dense_dyn_base(const derived_t& rhs) noexcept : base_type(rhs), _gpu(rhs._gpu) {
        //Nothing else to init
    }

    /*!
     * \brief Move construct a dense_dyn_base
     * \param rhs The dense_dyn_base to move from
     */
    dense_dyn_base(dense_dyn_base&& rhs) noexcept : base_type(std::move(rhs)), _gpu(std::move(rhs._gpu)) {
        //Nothing else to init
    }

    /*!
     * \brief Move construct a derived_t.
     *
     * This constructor is necessary in order to use the correct constructor in
     * the parent type.
     *
     * \param rhs The dense_dyn_base to move from
     */
    explicit dense_dyn_base(derived_t&& rhs) noexcept : base_type(std::move(rhs)), _gpu(std::move(rhs._gpu)) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a dense_dyn_base if the given size and dimensions
     * \param size The size of the matrix
     * \param dimensions The dimensions of the matrix
     */
    dense_dyn_base(size_t size, dimension_storage_impl dimensions) noexcept : base_type(size, dimensions) {
        //Nothing else to init
    }

    /*!
     * \brief Move construct a dense_dyn_base
     * \param rhs The dense_dyn_base to move from
     */
    template <typename E, cpp_enable_iff(!std::is_same<std::decay_t<E>, derived_t>::value)>
    explicit dense_dyn_base(E&& rhs) : base_type(std::move(rhs)) {
        //Nothing else to init
    }

    /*!
     * \brief Access the ith element of the matrix
     * \param i The index of the element to search
     * \return a reference to the ith element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 1, cpp_enable_iff(B)>
    value_type& operator()(size_t i) noexcept(assert_nothrow) {
        cpp_assert(i < dim(0), "Out of bounds");

        ensure_cpu_up_to_date();
        invalidate_gpu();

        return _memory[i];
    }

    /*!
     * \brief Access the ith element of the matrix
     * \param i The index of the element to search
     * \return a reference to the ith element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 1, cpp_enable_iff(B)>
    const value_type& operator()(size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < dim(0), "Out of bounds");

        ensure_cpu_up_to_date();

        return _memory[i];
    }

    /*!
     * \brief Returns the value at the position (sizes...)
     * \param sizes The indices
     * \return The value at the position (sizes...)
     */
    template <typename... S, cpp_enable_iff(
                                 (n_dimensions > 1) &&
                                 (sizeof...(S) == n_dimensions) &&
                                 cpp::all_convertible_to_v<size_t, S...>)>
    const value_type& operator()(S... sizes) const noexcept(assert_nothrow) {
        ensure_cpu_up_to_date();
        return _memory[etl::dyn_index(as_derived(), sizes...)];
    }

    /*!
     * \brief Returns the value at the position (sizes...)
     * \param sizes The indices
     * \return The value at the position (sizes...)
     */
    template <typename... S, cpp_enable_iff(
                                 (n_dimensions > 1) &&
                                 (sizeof...(S) == n_dimensions) &&
                                 cpp::all_convertible_to_v<size_t, S...>)>
    value_type& operator()(S... sizes) noexcept(assert_nothrow) {
        ensure_cpu_up_to_date();
        invalidate_gpu();
        return _memory[etl::dyn_index(as_derived(), sizes...)];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < _size, "Out of bounds");

        ensure_cpu_up_to_date();

        return _memory[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](size_t i) noexcept(assert_nothrow) {
        cpp_assert(i < _size, "Out of bounds");

        ensure_cpu_up_to_date();
        invalidate_gpu();

        return _memory[i];
    }

    /*!
     * \returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < _size, "Out of bounds");

        ensure_cpu_up_to_date();

        return _memory[i];
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_enable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_disable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return rhs.alias(as_derived());
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    inline memory_type memory_start() noexcept {
        return _memory;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    inline const_memory_type memory_start() const noexcept {
        return _memory;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer to the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) noexcept {
        return sub(as_derived(), i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) const noexcept {
        return sub(as_derived(), i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) noexcept {
        return etl::slice(as_derived(), first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) const noexcept {
        return etl::slice(as_derived(), first, last);
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    T* gpu_memory() const noexcept {
        return _gpu.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        _gpu.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        _gpu.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        _gpu.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        _gpu.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        _gpu.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        _gpu.ensure_gpu_allocated(_size);
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        _gpu.ensure_gpu_up_to_date(memory_start(), _size);
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        _gpu.ensure_cpu_up_to_date(memory_start(), _size);
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const T* gpu_memory) const {
        _gpu.gpu_copy_from(gpu_memory, _size);
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return _gpu.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return _gpu.is_gpu_up_to_date();
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }
};

} //end of namespace etl
