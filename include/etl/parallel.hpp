//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param n The size of the evaluation
 * \param threshold The parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool select_parallel(std::size_t n, std::size_t threshold = parallel_threshold) {
    return threads > 1 && ((parallel_support && local_context().parallel)|| (is_parallel && n >= threshold && !local_context().serial));
}

/*!
 * \brief Indicates if an 2D evaluation should run in paralle
 * \param n1 The first dimension of the evaluation
 * \param t1 The first parallel threshold
 * \param n2 The second dimension of the evaluation
 * \param t2 The second parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool select_parallel_2d(std::size_t n1, std::size_t t1, std::size_t n2, std::size_t t2) {
    return threads > 1 && ((parallel_support && local_context().parallel) || (is_parallel && n1 >= t1 && n2 >= t2 && !local_context().serial));
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner
 * \param pool The pool to use
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param T The number of threads to use
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename Functor>
inline void dispatch_1d(cpp::default_thread_pool<>& pool, bool p, Functor&& functor, size_t T, std::size_t first, std::size_t last) {
    if (p) {
        const auto n     = last - first;
        const auto batch = n / T;

        for (std::size_t t = 0; t < T - 1; ++t) {
            pool.do_task(functor, first + t * batch, first + (t + 1) * batch);
        }

        functor(first + (T - 1) * batch, last);

        pool.wait();
    } else {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner
 * \param pool The pool to use
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename Functor>
inline void dispatch_1d(cpp::default_thread_pool<>& pool, bool p, Functor&& functor, std::size_t first, std::size_t last) {
    if (p) {
        dispatch_1d(pool, p, std::forward<Functor>(functor), pool.size() + 1, first, last);
    } else {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename Functor>
inline void dispatch_1d(bool p, Functor&& functor, std::size_t first, std::size_t last) {
    if (p) {
        thread_local cpp::default_thread_pool<> pool(threads - 1);
        dispatch_1d(pool, p, std::forward<Functor>(functor), threads, first, last);
    } else {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename Functor>
inline void dispatch_1d_any(bool p, Functor&& functor, std::size_t first, std::size_t last) {
    if (p) {
        auto n = last - first;
        size_t T = std::min(n, threads);
        thread_local cpp::default_thread_pool<> pool(threads - 1);
        dispatch_1d(pool, p, std::forward<Functor>(functor), T, first, last);
    } else {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner.
 *
 * The dispatching will be done in batch. That is to say that the
 * functor will be called with a range of data.
 *
 * This will only be dispatched in parallel if etl is running in
 * parallel mode and if the range is bigger than the treshold.
 *
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range. Must be bigger or equal to first.
 * \param threshold The threshold for parallelization
 */
template <typename Functor>
inline void smart_dispatch_1d_any(Functor&& functor, size_t first, size_t last, size_t threshold) {
    if (etl::is_parallel) {
        auto n = last - first;

        if (select_parallel(n, threshold)) {
            size_t T = std::min(n, threads);
            thread_local cpp::default_thread_pool<> pool(threads - 1);
            dispatch_1d(pool, true, std::forward<Functor>(functor), T, first, last);
        } else {
            functor(first, last);
        }
    } else {
        functor(first, last);
    }
}

#ifdef ETL_PARALLEL_SUPPORT

/*!
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param n The size of the evaluation
 * \param threshold The parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool engine_select_parallel(std::size_t n, std::size_t threshold = parallel_threshold) {
    return threads > 1 && !local_context().serial && (local_context().parallel || (is_parallel && n >= threshold));
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine.
 *
 * The dispatching will be done in batch. That is to say that the
 * functor will be called with a range of data.
 *
 * This will only be dispatched in parallel if etl is running in
 * parallel mode and if the range is bigger than the treshold.
 *
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range. Must be bigger or equal to first.
 * \param threshold The threshold for parallelization
 */
template <typename Functor>
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, size_t threshold) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        if (engine_select_parallel(n, threshold)) {
            const size_t T     = std::min(n, etl::threads);
            const size_t batch = n / T;

            thread_engine::acquire();

            for (std::size_t t = 0; t < T - 1; ++t) {
                thread_engine::schedule(functor, first + t * batch, first + (t + 1) * batch);
            }

            thread_engine::schedule(functor, first + (T - 1) * batch, last);

            thread_engine::wait();
        } else {
            functor(first, last);
        }
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner and use an accumulator functor to accumulate the results
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename TT, typename Functor, typename AccFunctor>
inline void engine_dispatch_1d_acc(Functor&& functor, AccFunctor&& acc_functor, std::size_t first, std::size_t last, size_t threshold) {
    const size_t n     = last - first;

    if(n){
        if (engine_select_parallel(n, threshold)) {
            const size_t T     = std::min(n, etl::threads);
            const size_t batch = n / T;

            thread_engine::acquire();

            std::vector<TT> futures(T);

            auto sub_functor = [&futures, &functor](std::size_t t, std::size_t first, std::size_t last) {
                futures[t]   = functor(first, last);
            };

            for (size_t t = 0; t < T - 1; ++t) {
                thread_engine::schedule(sub_functor, t, first + t * batch, first + (t + 1) * batch);
            }

            thread_engine::schedule(sub_functor, T - 1, first + (T - 1) * batch, last);

            thread_engine::wait();

            for (auto fut : futures) {
                acc_functor(fut);
            }
        } else {
            acc_functor(functor(first, last));
        }
    }
}

#else

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine.
 *
 * The dispatching will be done in batch. That is to say that the
 * functor will be called with a range of data.
 *
 * This will only be dispatched in parallel if etl is running in
 * parallel mode and if the range is bigger than the treshold.
 *
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range. Must be bigger or equal to first.
 * \param threshold The threshold for parallelization
 */
template <typename Functor>
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, size_t threshold) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner and use an accumulator functor to accumulate the results
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename T, typename Functor, typename AccFunctor>
inline void engine_dispatch_1d_acc(Functor&& functor, AccFunctor&& acc_functor, std::size_t first, std::size_t last, size_t threshold) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        acc_functor(functor(first, last));
    }
}

#endif

} //end of namespace etl
