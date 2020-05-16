//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

#ifdef ETL_PARALLEL_SUPPORT

/*!
 * \brief The default thread engine.
 * \tparam Pool The thread pool implementation
 *
 * This should only be used by ETL internals such as the evaluator
 * and the engine_dispatch functions.
 */
template <typename Pool>
struct conf_thread_engine {
    /*!
     * \brief Acquire the thread engine.
     *
     * This function must be called before tasks are being
     * scheduled. It is mostly to ensure that selection is done
     * correctly and that the thread engine is used correclty.
     */
    static void acquire() {
        cpp_assert(etl::parallel_support, "thread_engine can only be used if paralle support is enabled");
        cpp_assert(!local_context().serial, "thread_engine cannot be used in serial context");
        cpp_assert(etl::threads > 1, "thread_engine cannot be used with less than 2");
        cpp_assert(is_parallel_session(), "thread_engine should only be used in parallel session");
    }

    /*!
     * \brief Schedule a new task
     * \param fun The functor to execute
     * \param args The arguments to pass to the functor
     */
    template <typename Functor, typename... Args>
    static void schedule(Functor&& fun, Args&&... args) {
        get_pool().do_task(std::forward<Functor>(fun), std::forward<Args>(args)...);
    }

    /*!
     * \brief Wait for all the scheduled threads to finish their task
     */
    static void wait() {
        get_pool().wait();
    }

private:
    /*!
     * \brief Returns a reference to the thread pool
     * \return The unique thread pool.
     */
    static Pool& get_pool() {
        static Pool pool(etl::threads);
        return pool;
    }
};

using thread_engine = conf_thread_engine<cpp::default_thread_pool<>>;

#else

/*!
 * \brief The default thread engine when auto-parallelization is not enabled
 *
 * This should only be used by ETL internals such as the evaluator
 * and the engine_dispatch functions.
 */
struct thread_engine {
    /*!
     * \brief Acquire the thread engine.
     *
     * This function must be called before tasks are being
     * scheduled. It is mostly to ensure that selection is done
     * correctly and that the thread engine is used correclty.
     */
    static void acquire() {
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }

    /*!
     * \brief Schedule a new task
     * \param fun The functor to execute
     */
    template <typename Functor, typename... Args>
    static void schedule([[maybe_unused]] Functor&& fun, [[maybe_unused]] Args&&... args) {
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }

    /*!
     * \brief Wait for all the scheduled threads to finish their task
     */
    static void wait() {
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }
};

#endif

} //end of namespace etl
