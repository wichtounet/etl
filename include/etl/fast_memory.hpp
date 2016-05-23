//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template <typename T, std::size_t D, order SO>
struct fast_memory {
    static constexpr const std::size_t n_dimensions = D;                      ///< The number of dimensions
    static constexpr const order storage_order      = SO;                                   ///< The storage order

    using value_type        = T;
    using memory_type       = T*;
    using const_memory_type = std::add_const_t<T>*;

    T* memory;
    const std::size_t etl_size;
    const std::array<std::size_t, D> dims;

    fast_memory(T* memory, std::size_t size, const std::array<std::size_t, D>& dims) : memory(memory), etl_size(size), dims(dims) {
        //Nothing else to init
    }

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
    }

    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    std::size_t size() noexcept {
        return etl_size;
    }

    /*!
     * \brief Returns the Dth dimension of the matrix
     * \return The Dth dimension of the matrix
     */
    template <std::size_t DD>
    std::size_t dim() noexcept {
        return dims[DD];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return &memory[0];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return &memory[0];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return &memory[size()];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return &memory[size()];
    }
};

} //end of namespace etl
