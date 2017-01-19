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
    return threads > 1 && (local_context().parallel || (is_parallel && n >= threshold && !local_context().serial));
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
    return threads > 1 && (local_context().parallel || (is_parallel && n1 >= t1 && n2 >= t2 && !local_context().serial));
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
        dispatch_1d(pool, p, std::forward<Functor>(functor), pool.size(), first, last);
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
        dispatch_1d(pool, p, std::forward<Functor>(functor), first, last);
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
        thread_local cpp::default_thread_pool<> pool(threads);
        dispatch_1d(pool, p, std::forward<Functor>(functor), T, first, last);
    } else {
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
inline void dispatch_1d_acc(bool p, Functor&& functor, AccFunctor&& acc_functor, std::size_t first, std::size_t last) {
    if (p) {
        std::vector<T> futures(threads - 1);
        thread_local cpp::default_thread_pool<> pool(threads - 1);

        auto n     = last - first;
        auto batch = n / threads;

        auto sub_functor = [&futures, &functor](std::size_t t, std::size_t first, std::size_t last) {
            futures[t]   = functor(first, last);
        };

        for (std::size_t t = 0; t < threads - 1; ++t) {
            pool.do_task(sub_functor, t, first + t * batch, first + (t + 1) * batch);
        }

        acc_functor(functor(first + (threads - 1) * batch, last));

        pool.wait();

        for (auto fut : futures) {
            acc_functor(fut);
        }
    } else {
        acc_functor(functor(first, last));
    }
}

} //end of namespace etl
