//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDA
#include "etl/impl/cublas/cuda.hpp"
#endif

namespace etl {

#ifdef ETL_CUDA

/*!
 * \brief GPU memory allocator.
 *
 * All GPU memory allocations should be made through this helper.
 */
struct gpu_memory_allocator {
private:
    /*!
     * \brief Allocate GPU memory of the given size
     * \param size The memory  size
     * \return The allocated GPU memory
     */
    template <typename T>
    static T* base_allocate(size_t size) {
        T* memory = nullptr;

        auto cuda_status = cudaMalloc(&memory, size * sizeof(T));

        if (cuda_status != cudaSuccess) {
            std::cout << "cuda: Failed to allocate GPU memory: " << cudaGetErrorString(cuda_status) << std::endl;
            std::cout << "      Tried to allocate " << size * sizeof(T) << "B" << std::endl;
            exit(EXIT_FAILURE);
        }

        inc_counter("gpu:allocate");

        return memory;
    }

    /*!
     * \brief Release memory allocated by this allocator
     * \param gpu_memory The GPU memory allocated
     * \param size The size of the allocated GPU memory
     */
    static void base_release(const void* gpu_memory) {
        //Note: the const_cast is only here to allow compilation
        cuda_check(cudaFree((const_cast<void*>(gpu_memory))));

        inc_counter("gpu:release");
    }

#ifndef ETL_GPU_POOL
public:
    /*!
     * \brief Allocate GPU memory of the given size
     * \param size The memory  size
     * \return The allocated GPU memory
     */
    template <typename T>
    static T* allocate(size_t size) {
        return base_allocate<T>(size);
    }

    /*!
     * \brief Release memory allocated by this allocator
     * \param gpu_memory The GPU memory allocated
     * \param size The size of the allocated GPU memory
     */
    static void release(const void* gpu_memory, [[maybe_unused]] size_t size) {
        base_release(gpu_memory);
    }

    /*!
     * \brief Release all memory acquired by the allocator.
     *
     * This has no effect if the allocator does not use a memory
     * pool.
     */
    static void clear() {
        // This allocator does not store memory
    }

#else // ETL_GPU_POOL

#ifdef ETL_GPU_POOL_SIZE
    static constexpr size_t entries = ETL_GPU_POOL_SIZE;  ///< The entries limit of the pool
#else
#ifdef ETL_GPU_POOL_LIMIT
    static constexpr size_t entries = 256;                ///< The entries limit of the pool
#else
    static constexpr size_t entries = 64; ///< The entries limit of the pool
#endif
#endif

#ifdef ETL_GPU_POOL_LIMIT
    static constexpr size_t limit   = ETL_GPU_POOL_LIMIT; ///< The size limit of the pool
#else
    static constexpr size_t limit   = 1024 * 1024 * 1024; ///< The size limit of the pool
#endif

    /*!
     * \brief An entry in the mini-pool
     */
    struct mini_pool_entry {
        size_t size  = 0;       ///< The size, in bytes, of the memory
        void* memory = nullptr; ///< Pointer to the memory, if any
    };

    /*!
     * \brief A very simple implementation for a GPU memory pool of
     * fixed-size
     *
     * Although it's quite simple, it should help in most cases
     * where the same operations are done several times, as in
     * Machine Learning. In that case, even a very simple pool like
     * this may greatly reduce the overhead of memory allocation.
     */
    struct mini_pool {
        std::array<mini_pool_entry, entries> cache; ///< The cache of GPU memory addresses
    };

    /*!
     * \brief Return a reference to the GPU pool
     * \return a reference to the GPU pool
     */
    static mini_pool& get_pool() {
        static mini_pool pool;
        return pool;
    }

    /*!
     * \brief Return a reference to the GPU pool size
     * \return a reference to the GPU pool size
     */
    static size_t& get_pool_size() {
        static size_t pool_size = 0;
        return pool_size;
    }

    /*!
     * \brief Return the lock for the pool
     * \return a reference to the pool lock
     */
    static std::mutex& get_lock() {
        static std::mutex lock;
        return lock;
    }

public:
    /*!
     * \brief Allocate GPU memory of the given size
     * \param size The memory  size
     * \return The allocated GPU memory
     */
    template <typename T>
    static T* allocate(size_t size) {
        const auto real_size = size * sizeof(T);

        // Try to get memory from the pool

        {
            std::lock_guard<std::mutex> l(get_lock());

            if (get_pool_size()) {
                for (auto& slot : get_pool().cache) {
                    if (slot.memory && slot.size == real_size) {
                        auto memory = slot.memory;
                        slot.memory = nullptr;

                        get_pool_size() -= size;

                        return static_cast<T*>(memory);
                    }
                }
            }
        }

        // If a memory block is not found, allocate new memory

        return base_allocate<T>(size);
    }

    /*!
     * \brief Release memory allocated by this allocator
     * \param gpu_memory The GPU memory allocated
     * \param size The size of the allocated GPU memory
     */
    template <typename T>
    static void release(const T* gpu_memory, size_t size) {
        // Try to get an empty slot

        {
            std::lock_guard<std::mutex> l(get_lock());

            if (get_pool_size() + size < limit) {
                for (auto& slot : get_pool().cache) {
                    if (!slot.memory) {
                        slot.memory = const_cast<void*>(static_cast<const void*>(gpu_memory));
                        slot.size   = size * sizeof(T);

                        get_pool_size() += size;

                        return;
                    }
                }
            }
        }

        // If the cache is full, release the memory

        base_release(gpu_memory);
    }

    /*!
     * \brief Release all memory acquired by the allocator.
     *
     * This has no effect if the allocator does not use a memory
     * pool.
     */
    static void clear() {
        std::lock_guard<std::mutex> l(get_lock());

        // Release each used slots
        // and clear them

        for (auto& slot : get_pool().cache) {
            if (slot.memory) {
                base_release(slot.memory);

                slot.memory = nullptr;
                slot.size   = 0;
            }
        }

        get_pool_size() = 0;
    }
#endif
};

/*!
 * \brief GPU memory handler.
 *
 * This handler is responsible for allocating the memory and keeping CPU and GPU
 * memory consistency.
 */
template <typename T>
struct gpu_memory_handler {
private:
    mutable T* gpu_memory_         = nullptr; ///< The GPU memory pointer
    mutable size_t gpu_memory_size = 0;       ///< The GPU memory size

    mutable bool cpu_up_to_date = true;  ///< Is the CPU memory up to date
    mutable bool gpu_up_to_date = false; ///< Is the GPU memory up to date

public:
    gpu_memory_handler() = default;

    /*!
     * \brief Copy construct a gpu_memory_handler
     * \param the gpu_memory_handler to copy from
     */
    gpu_memory_handler(const gpu_memory_handler& rhs)
            : gpu_memory_size(rhs.gpu_memory_size), cpu_up_to_date(rhs.cpu_up_to_date), gpu_up_to_date(rhs.gpu_up_to_date) {
        if (rhs.gpu_up_to_date) {
            gpu_allocate_impl(gpu_memory_size);

            gpu_copy_from(rhs.gpu_memory_, gpu_memory_size);

            // The CPU status can be erased by gpu_copy_from
            if (rhs.cpu_up_to_date) {
                validate_cpu();
            }
        } else {
            gpu_memory_ = nullptr;
        }

        cpp_assert(rhs.is_cpu_up_to_date() == this->is_cpu_up_to_date(), "gpu_memory_handler(&) must preserve CPU status");
        cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "gpu_memory_handler(&) must preserve GPU status");
    }

    /*!
     * \brief Move construct a gpu_memory_handler
     */
    gpu_memory_handler(gpu_memory_handler&& rhs) noexcept
            : gpu_memory_(rhs.gpu_memory_), gpu_memory_size(rhs.gpu_memory_size), cpu_up_to_date(rhs.cpu_up_to_date), gpu_up_to_date(rhs.gpu_up_to_date) {
        rhs.gpu_memory_     = nullptr;
        rhs.gpu_memory_size = 0;
    }

    /*!
     * \brief Copy assign a gpu_memory_handler
     * \param the gpu_memory_handler to copy from
     * \return a reference to this object
     */
    gpu_memory_handler& operator=(const gpu_memory_handler& rhs) {
        if (this != &rhs) {
            // Release the previous memory, if any
            if (gpu_memory_) {
                gpu_memory_allocator::release(gpu_memory_, gpu_memory_size);
                gpu_memory_ = nullptr;
            }

            // Copy the size from rhs
            gpu_memory_size = rhs.gpu_memory_size;

            // Copy the contents of rhs
            if (rhs.gpu_up_to_date) {
                gpu_allocate_impl(gpu_memory_size);

                gpu_copy_from(rhs.gpu_memory_, gpu_memory_size);
            } else {
                gpu_memory_ = nullptr;
            }

            // Copy the status (at the end, otherwise gpu_copy_from will screw them)
            cpu_up_to_date = rhs.cpu_up_to_date;
            gpu_up_to_date = rhs.gpu_up_to_date;
        }

        return *this;
    }

    /*!
     * \brief Move assign a gpu_memory_handler
     */
    gpu_memory_handler& operator=(gpu_memory_handler&& rhs) noexcept {
        if (this != &rhs) {
            // Release the previous memory, if any
            if (gpu_memory_) {
                gpu_memory_allocator::release(gpu_memory_, gpu_memory_size);
                gpu_memory_ = nullptr;
            }

            // Steal the values and contents from rhs
            gpu_memory_     = rhs.gpu_memory_;
            gpu_memory_size = rhs.gpu_memory_size;
            cpu_up_to_date  = rhs.cpu_up_to_date;
            gpu_up_to_date  = rhs.gpu_up_to_date;

            // Make sure rhs does not have point to the memory
            rhs.gpu_memory_     = nullptr;
            rhs.gpu_memory_size = 0;
        }

        return *this;
    }

    /*!
     * \brief Destroys the GPU memory handler. This effectively
     * releases any memory allocated.
     */
    ~gpu_memory_handler() {
        if (gpu_memory_) {
            gpu_memory_allocator::release(gpu_memory_, gpu_memory_size);
        }
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return cpu_up_to_date;
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return gpu_up_to_date;
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    T* gpu_memory() const noexcept {
        return gpu_memory_;
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if (gpu_memory_) {
            gpu_memory_allocator::release(gpu_memory_, gpu_memory_size);

            gpu_memory_     = nullptr;
            gpu_memory_size = 0;
        }

        invalidate_gpu();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        cpu_up_to_date = false;

        cpp_assert(gpu_up_to_date, "Cannot invalidate the CPU if the GPU is not up to date");
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        gpu_up_to_date = false;

        cpp_assert(cpu_up_to_date, "Cannot invalidate the GPU if the CPU is not up to date");
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        cpu_up_to_date = true;
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        gpu_up_to_date = true;
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     * \param etl_size The size of the memory
     */
    void ensure_gpu_allocated(size_t etl_size) const {
        if (!is_gpu_allocated()) {
            gpu_allocate_impl(etl_size);
        }
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     * \param cpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void ensure_gpu_up_to_date(const T* cpu_memory, size_t etl_size) const {
        // Make sure there is some memory allocate
        if (!is_gpu_allocated()) {
            gpu_allocate_impl(etl_size);
        }

        if (!gpu_up_to_date) {
            cpu_to_gpu(cpu_memory, etl_size);
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     * \param cpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void ensure_cpu_up_to_date(const T* cpu_memory, size_t etl_size) const {
        if (!cpu_up_to_date) {
            gpu_to_cpu(cpu_memory, etl_size);
        }
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void gpu_copy_from(const T* gpu_memory, size_t etl_size) const {
        cpp_assert(is_gpu_allocated(), "GPU must be allocated before copy");
        cpp_assert(gpu_memory, "Cannot copy from invalid memory");
        cpp_assert(etl_size, "Cannot copy with a size of zero");

        cuda_check(cudaMemcpy(const_cast<std::remove_const_t<T>*>(gpu_memory_), const_cast<std::remove_const_t<T>*>(gpu_memory), etl_size * sizeof(T),
                              cudaMemcpyDeviceToDevice));

        gpu_up_to_date = true;
        cpu_up_to_date = false;
    }

private:
    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate_impl(size_t etl_size) const {
        cpp_assert(!is_gpu_allocated(), "Trying to allocate already allocated GPU gpu_memory_");

        gpu_memory_     = gpu_memory_allocator::allocate<T>(etl_size);
        gpu_memory_size = etl_size;
    }

    /*!
     * \brief Copy back from the CPU to the GPU
     */
    void cpu_to_gpu(const T* cpu_memory, size_t etl_size) const {
        cpp_assert(is_gpu_allocated(), "Cannot copy to unallocated GPU memory");
        cpp_assert(!gpu_up_to_date, "Copy must only be done if necessary");
        cpp_assert(cpu_up_to_date, "Copy from invalid memory!");
        cpp_assert(cpu_memory, "cpu_memory is nullptr in entry to cpu_to_gpu");
        cpp_assert(gpu_memory_, "gpu_memory_ is nullptr in entry to cpu_to_gpu");

        cuda_check(cudaMemcpy(const_cast<std::remove_const_t<T>*>(gpu_memory_), const_cast<std::remove_const_t<T>*>(cpu_memory), etl_size * sizeof(T),
                              cudaMemcpyHostToDevice));

        gpu_up_to_date = true;

        inc_counter("gpu:cpu_to_gpu");
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_to_cpu(const T* cpu_memory, size_t etl_size) const {
        cpp_assert(is_gpu_allocated(), "Cannot copy from unallocated GPU memory()");
        cpp_assert(gpu_up_to_date, "Cannot copy from invalid memory");
        cpp_assert(!cpu_up_to_date, "Copy done without reason");
        cpp_assert(cpu_memory, "cpu_memory is nullptr in entry to gpu_to_cpu");
        cpp_assert(gpu_memory_, "gpu_memory_ is nullptr in entry to gpu_to_cpu");

        cuda_check(cudaMemcpy(const_cast<std::remove_const_t<T>*>(cpu_memory), const_cast<std::remove_const_t<T>*>(gpu_memory_), etl_size * sizeof(T),
                              cudaMemcpyDeviceToHost));

        cpu_up_to_date = true;

        inc_counter("gpu:gpu_to_cpu");
    }

    /*!
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return gpu_memory_;
    }
};

#else
template <typename T>
struct gpu_memory_handler {
    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    T* gpu_memory() const noexcept {
        return nullptr;
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return true;
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return false;
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {}

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {}

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {}

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {}

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {}

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     * \param etl_size The size of the memory
     */
    void ensure_gpu_allocated([[maybe_unused]] size_t etl_size) const {}

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     * \param cpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void ensure_gpu_up_to_date([[maybe_unused]] const T* cpu_memory, [[maybe_unused]] size_t etl_size) const {}

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     * \param cpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void ensure_cpu_up_to_date([[maybe_unused]] const T* cpu_memory, [[maybe_unused]] size_t etl_size) const {}

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     * \param etl_size The size of the memory
     */
    void gpu_copy_from([[maybe_unused]] const T* gpu_memory, [[maybe_unused]] size_t etl_size) const {}
};
#endif

} //end of namespace etl

#ifdef ETL_CUDA
#include "etl/impl/cublas/cuda_memory.hpp"
#endif
