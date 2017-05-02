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
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param select The secondary parallel selection
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool engine_select_parallel(bool select) {
    return threads > 1 && !local_context().serial && (local_context().parallel || select);
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

inline std::pair<size_t, size_t> thread_blocks(size_t M, size_t N) {
    if (M >= N) {
        size_t m = std::min(threads, std::max(1UL, size_t(round(std::sqrt(threads * double(M) / double(N))))));
        size_t n = threads / m;

        while (m * n != threads) {
            ++m;
            n = threads / m;
        }

        return {m, n};
    } else {
        size_t n = std::min(threads, std::max(1UL, size_t(round(std::sqrt(threads * double(N) / double(M))))));
        size_t m = threads / n;

        while (m * n != threads) {
            ++n;
            m = threads / n;
        }

        return {m, n};
    }
}

/*!
 * \brief Dispatch the elements of a 2D range to a functor in a parallel
 * manner, using the global thread engine.
 *
 * The dispatching will be done in batch. That is to say that the
 * functor will be called with a range of data.
 *
 * This will only be dispatched in parallel if etl is running in
 * parallel mode and if the range is bigger than the treshold.
 *
 * \param functor The functor to execute
 * \param last1 The size of the first range
 * \param last2 The size of the first range
 * \param threshold The threshold for parallelization
 */
template <typename Functor>
inline void engine_dispatch_2d(Functor&& functor, size_t last1, size_t last2, size_t threshold) {
    if (last1 && last2) {
        if (engine_select_parallel(last1 * last2, threshold)) {
            thread_engine::acquire();

            auto block = thread_blocks(last1, last2);

            const size_t block_1 = last1 / block.first + (last1 % block.first > 0);
            const size_t block_2 = last2 / block.second + (last2 % block.second > 0);

            for(size_t i = 0; i < block.first; ++i){
                const size_t row = block_1 * i;

                if(cpp_unlikely(row >= last1)){
                    continue;
                }

                for(size_t j = 0; j < block.second; ++j){
                    const size_t column = block_2 * j;

                    if(cpp_unlikely(column >= last2)){
                        continue;
                    }

                    const size_t m = std::min(block_1, last1 - row);
                    const size_t n = std::min(block_2, last2 - column);

                    thread_engine::schedule(functor, row, row + m, column, column + n);
                }
            }

            thread_engine::wait();
        } else {
            functor(0, last1, 0, last2);
        }
    }
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
 * \param select The selector for parallelization
 */
template <typename Functor>
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, bool select) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        if (engine_select_parallel(select)) {
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
 *
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param first The beginning of the range
 * \param last The end of the range
 * \param threshold The threshold for parallelization
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

    cpp_unused(threshold);

    const size_t n = last - first;

    if (n) {
        functor(first, last);
    }
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
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, bool select) {
    cpp_assert(last >= first, "Range must be valid");

    cpp_unused(select);

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

    cpp_unused(threshold);

    const size_t n = last - first;

    if (n) {
        acc_functor(functor(first, last));
    }
}

#endif

} //end of namespace etl
