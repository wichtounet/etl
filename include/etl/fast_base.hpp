//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains static matrix implementation
 */

#pragma once

#include "etl/index.hpp"

namespace etl {

namespace matrix_detail {

/*!
 * \brief Traits to test if a type is a std::vector
 */
template <typename N>
struct is_vector_impl : std::false_type {};

/*!
 * \copydoc is_vector
 */
template <typename... Args>
struct is_vector_impl<std::vector<Args...>> : std::true_type {};

/*!
 * \copydoc is_vector
 */
template <typename... Args>
struct is_vector_impl<etl::simple_vector<Args...>> : std::true_type {};

/*!
 * \brief Traits to test if a type is a std::vector
 */
template <typename N>
constexpr bool is_vector = is_vector_impl<N>::value;

/*!
 * \brief Traits to extract iterator types from a type
 */
template <typename T>
struct iterator_type {
    using iterator       = typename T::iterator;       ///< The iterator type
    using const_iterator = typename T::const_iterator; ///< The const iterator type
};

/*!
 * \copydoc iterator_type
 */
template <typename T>
struct iterator_type<T*> {
    using iterator       = T*;       ///< The iterator type
    using const_iterator = const T*; ///< The const iterator type
};

/*!
 * \copydoc iterator_type
 */
template <typename T>
struct iterator_type<const T*> {
    using iterator       = const T*; ///< The iterator type
    using const_iterator = const T*; ///< The const iterator type
};

/*!
 * \brief Helper to get the iterator type from a type
 */
template <typename T>
using iterator_t = typename iterator_type<T>::iterator;

/*!
 * \brief Helper to get the const iterator type from a type
 */
template <typename T>
using const_iterator_t = typename iterator_type<T>::const_iterator;

} //end of namespace matrix_detail

template <typename D, typename T, typename ST, order SO, size_t... Dims>
struct fast_matrix_base {
    static constexpr size_t n_dimensions = sizeof...(Dims); ///< The number of dimensions
    static constexpr size_t etl_size     = (Dims * ...);    ///< The size of the matrix

    using value_type        = T;                 ///< The type of value
    using derived_t         = D;                 ///< The derived type
    using storage_impl      = ST;                ///< The storage implementation
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type

    using iterator = std::conditional_t<SO == order::RowMajor, value_type*, etl::iterator<derived_t>>;             ///< The iterator type
    using const_iterator = std::conditional_t<SO == order::RowMajor, const value_type*, etl::iterator<const derived_t>>; ///< The iterator type

protected:
    storage_impl _data;         ///< The storage container
    gpu_memory_handler<T> _gpu; ///< The GPU memory handler

    /*!
     * \brief Init the container if necessary
     */
    template <typename S = ST, cpp_enable_iff(matrix_detail::is_vector<S>)>
    void init() {
        _data.resize(alloc_size_mat<value_type>(size(), dim<n_dimensions - 1>()));
    }

    /*!
     * \copydoc init
     */
    template <typename S = ST, cpp_disable_iff(matrix_detail::is_vector<S>)>
    void init() noexcept {
        //Nothing else to init
    }

    /*!
     * \brief Compute the 1D index from the given indices
     * \param args The access indices
     * \return The 1D index inside the storage container
     */
    template <typename... S>
    static constexpr size_t index(S... args) {
        return etl::fast_index<derived_t>(args...);
    }

    /*!
     * \brief Return the value at the given indices
     * \param args The access indices
     * \return The value at the given indices
     */
    template <typename... S>
    value_type& access(S... args) {
        ensure_cpu_up_to_date();
        invalidate_gpu();
        return _data[index(args...)];
    }

    /*!
     * \brief Return the value at the given indices
     * \param args The access indices
     * \return The value at the given indices
     */
    template <typename... S>
    const value_type& access(S... args) const {
        ensure_cpu_up_to_date();
        return _data[index(args...)];
    }

public:
    /*!
     * \brief Construct a default fast_matrix_base
     */
    fast_matrix_base() : _data() {
        init();
    }

    /*!
     * \brief Construct a default fast_matrix_base from storage
     * \param data The storage data
     */
    fast_matrix_base(storage_impl data) : _data(data) {
        // Nothing else to init
    }

    /*!
     * \brief Copy Construct a fast_matrix_base
     * \param rhs The right hand side
     */
    fast_matrix_base(const fast_matrix_base& rhs) : _data(), _gpu(rhs._gpu) {
        // Only perform the copy if the CPU is up to date
        if(rhs._gpu.is_cpu_up_to_date()){
            _data = rhs._data;
        } else {
            init();
        }

        // The GPU updates are handled by gpu_handler
    }

    /*!
     * \brief Move Construct a fast_matrix_base
     * \param rhs The right hand side
     */
    fast_matrix_base(fast_matrix_base&& rhs) noexcept : _data(), _gpu(std::move(rhs._gpu)) {
        // Only perform the copy if the CPU is up to date
        if(_gpu.is_cpu_up_to_date()){
            _data = std::move(rhs._data);
        } else {
            init();
        }

        // The GPU updates are handled by gpu_handler
    }

    // Assignment must be handled by the derived class!

    fast_matrix_base& operator=(const fast_matrix_base& rhs) = delete;
    fast_matrix_base& operator=(fast_matrix_base&& rhs) = delete;

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return &_data[0];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return &_data[0];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return &_data[size()];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return &_data[size()];
    }

    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    static constexpr size_t size() noexcept {
        return etl_size;
    }

    /*!
     * \brief Returns the number of rows of the matrix (the first dimension), in O(1)
     * \return The number of rows of the matrix
     */
    static constexpr size_t rows() noexcept {
        return dim<0>();
    }

    /*!
     * \brief Returns the number of columns of the matrix (the second dimension), in O(1)
     * \return The number of columns of the matrix
     */
    static constexpr size_t columns() noexcept {
        static_assert(n_dimensions > 1, "columns() can only be used on 2D+ matrices");

        return dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr size_t dimensions() noexcept {
        return n_dimensions;
    }

    /*!
     * \brief Returns the Dth dimension of the matrix
     * \return The Dth dimension of the matrix
     */
    template <size_t DD>
    static constexpr size_t dim() noexcept {
        return nth_size<DD, 0, Dims...>;
    }

    /*!
     * \brief Returns the dth dimension of the matrix
     * \param d The dimension to get
     * \return The Dth dimension of the matrix
     */
    size_t dim(size_t d) const noexcept {
        return dyn_nth_size<Dims...>(d);
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
     * \brief Returns the value of the element at the position (args...)
     * \param args The position indices
     * \return The value of the element at (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    value_type& operator()(S... args) noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return access(static_cast<size_t>(args)...);
    }

    /*!
     * \brief Returns the value of the element at the position (args...)
     * \param args The position indices
     * \return The value of the element at (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    const value_type& operator()(S... args) const noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return access(static_cast<size_t>(args)...);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        ensure_cpu_up_to_date();

        return _data[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](size_t i) noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        ensure_cpu_up_to_date();
        invalidate_gpu();

        return _data[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        ensure_cpu_up_to_date();

        return _data[i];
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_enable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_disable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return rhs.alias(as_derived());
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
        _gpu.ensure_gpu_allocated(etl_size);
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        _gpu.ensure_gpu_up_to_date(memory_start(), etl_size);
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        _gpu.ensure_cpu_up_to_date(memory_start(), etl_size);
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const T* gpu_memory) const {
        _gpu.gpu_copy_from(gpu_memory, etl_size);
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
