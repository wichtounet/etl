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
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param n The size of the evaluation
 * \param threshold The parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool engine_select_parallel(size_t n, size_t threshold = parallel_threshold) {
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

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                for (size_t t = 0; t < T - 1; ++t) {
                    thread_engine::schedule(functor, first + t * batch, first + (t + 1) * batch);
                }

                thread_engine::schedule(functor, first + (T - 1) * batch, last);

                thread_engine::wait();
            }
        } else {
            functor(first, last);
        }
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned thread will
 * be prevented from opening new threads, by using a serial section.
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
inline void engine_dispatch_1d_serial(Functor&& functor, size_t first, size_t last, size_t threshold) {
    auto serial_functor = [&functor](size_t first, size_t last) {
        SERIAL_SECTION {
            functor(first, last);
        }
    };

    engine_dispatch_1d(serial_functor, first, last, threshold);
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned thread will
 * be prevented from using the GPU, by using a CPU-only section.
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
inline void engine_dispatch_1d_cpu(Functor&& functor, size_t first, size_t last, size_t threshold) {
    auto cpu_functor = [&functor](size_t first, size_t last) {
        CPU_SECTION {
            functor(first, last);
        }
    };

    engine_dispatch_1d(cpu_functor, first, last, threshold);
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned thread will
 * be prevented from opening new threads or using the GPU, by
 * using a serial section and a CPU-only section.
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
inline void engine_dispatch_1d_serial_cpu(Functor&& functor, size_t first, size_t last, size_t threshold) {
    auto serial_cpu_functor = [&functor](size_t first, size_t last) {
        SERIAL_SECTION {
            CPU_SECTION {
                functor(first, last);
            }
        }
    };

    engine_dispatch_1d(serial_cpu_functor, first, last, threshold);
}

/*!
 * \brief Split a matrix into blocks for each thread for parallel
 * dispatching
 * \param M The first dimension of the matrix
 * \param N The second dimension of the matrix
 * \return A pair containing the width and height of the blocks
 * processed by each thread
 */
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
            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                auto[blocks1, blocks2] = thread_blocks(last1, last2);

                const size_t block_1 = last1 / blocks1 + (last1 % blocks1 > 0);
                const size_t block_2 = last2 / blocks2 + (last2 % blocks2 > 0);

                for (size_t i = 0; i < blocks1; ++i) {
                    const size_t row = block_1 * i;

                    if (cpp_unlikely(row >= last1)) {
                        continue;
                    }

                    for (size_t j = 0; j < blocks2; ++j) {
                        const size_t column = block_2 * j;

                        if (cpp_unlikely(column >= last2)) {
                            continue;
                        }

                        const size_t m = std::min(block_1, last1 - row);
                        const size_t n = std::min(block_2, last2 - column);

                        thread_engine::schedule(functor, row, row + m, column, column + n);
                    }
                }

                thread_engine::wait();
            }
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

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                for (size_t t = 0; t < T - 1; ++t) {
                    thread_engine::schedule(functor, first + t * batch, first + (t + 1) * batch);
                }

                thread_engine::schedule(functor, first + (T - 1) * batch, last);

                thread_engine::wait();
            }
        } else {
            functor(first, last);
        }
    }
}

template <typename Functor>
inline void engine_dispatch_1d_block(Functor&& functor, size_t first, size_t last, size_t block) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        if (engine_select_parallel(n >= 2 * block)) {
            const size_t T     = std::min(n / block, etl::threads);
            const size_t batch = n / (T * block);

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                for (size_t t = 0; t < T - 1; ++t) {
                    thread_engine::schedule(functor, first + t * batch, first + (t + 1) * batch);
                }

                thread_engine::schedule(functor, first + (T - 1) * batch, last);

                thread_engine::wait();
            }
        } else {
            functor(first, last);
        }
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned threads will
 * be prevented from opening new threads by constraining the functor
 * into a serial section.
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
inline void engine_dispatch_1d_serial(Functor&& functor, size_t first, size_t last, bool select) {
    auto serial_functor = [&functor](size_t first, size_t last) {
        SERIAL_SECTION {
            functor(first, last);
        }
    };

    engine_dispatch_1d(serial_functor, first, last, select);
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned threads will
 * be prevented from opening new threads and using the GPU by
 * constraining the functor into a serial section and a CPU-only section.
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
inline void engine_dispatch_1d_serial_cpu(Functor&& functor, size_t first, size_t last, bool select) {
    auto serial_cpu_functor = [&functor](size_t first, size_t last) {
        SERIAL_SECTION {
            CPU_SECTION {
                functor(first, last);
            }
        }
    };

    engine_dispatch_1d(serial_cpu_functor, first, last, select);
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel
 * manner, using the global thread engine. The spawned threads will
 * be prevented from using the GPU by constraining the functor
 * into a CPU-only section.
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
inline void engine_dispatch_1d_cpu(Functor&& functor, size_t first, size_t last, bool select) {
    auto cpu_functor = [&functor](size_t first, size_t last) {
        CPU_SECTION {
            functor(first, last);
        }
    };

    engine_dispatch_1d(cpu_functor, first, last, select);
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
inline void engine_dispatch_1d_acc(Functor&& functor, AccFunctor&& acc_functor, size_t first, size_t last, size_t threshold) {
    const size_t n = last - first;

    if (n) {
        if (engine_select_parallel(n, threshold)) {
            const size_t T     = std::min(n, etl::threads);
            const size_t batch = n / T;

            std::vector<TT> futures(T);

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                auto sub_functor = [&futures, &functor](size_t t, size_t first, size_t last) { futures[t] = functor(first, last); };

                for (size_t t = 0; t < T - 1; ++t) {
                    thread_engine::schedule(sub_functor, t, first + t * batch, first + (t + 1) * batch);
                }

                thread_engine::schedule(sub_functor, T - 1, first + (T - 1) * batch, last);

                thread_engine::wait();
            }

            for (auto fut : futures) {
                acc_functor(fut);
            }
        } else {
            acc_functor(functor(first, last));
        }
    }
}

/*!
 * \brief Dispatch the elements of an ETL container in a parallel manner and use an accumulator functor to accumulate the results.
 *
 * The functors will be called with slices of the original expression.
 *
 * \param expr The expression to slice
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param threshold The threshold for paralellization
 */
template <typename E, typename Functor>
inline void engine_dispatch_1d_slice(E&& expr, Functor&& functor, size_t threshold) {
    using TT = value_t<E>;

    static constexpr size_t S = default_intrinsic_traits<TT>::size;

    const size_t n = etl::size(expr);

    if (n) {
        if (engine_select_parallel(n, threshold)) {
            const size_t T = std::min(n, etl::threads);

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                if constexpr (decay_traits<E>::is_aligned && S > 1) {
                    if (n >= T * S) {
                        // In case there is enough data, we align it

                        const size_t n_aligned         = (n + (S - 1)) & ~(S - 1);
                        const size_t blocks            = n_aligned / S;
                        const size_t blocks_per_thread = blocks / T;
                        const size_t batch             = blocks_per_thread * S;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(functor, memory_slice<aligned>(expr, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(functor, memory_slice<aligned>(expr, (T - 1) * batch, n));
                    } else {
                        // Not enough data to consider aligning

                        const size_t batch = n / T;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(functor, memory_slice<unaligned>(expr, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(functor, memory_slice<unaligned>(expr, (T - 1) * batch, n));
                    }
                } else {
                    // If the data is not aligned in the first, don't make any effort to align it

                    const size_t batch = n / T;

                    for (size_t t = 0; t < T - 1; ++t) {
                        thread_engine::schedule(functor, memory_slice<unaligned>(expr, t * batch, (t + 1) * batch));
                    }

                    thread_engine::schedule(functor, memory_slice<unaligned>(expr, (T - 1) * batch, n));
                }

                thread_engine::wait();
            }
        } else {
            functor(expr);
        }
    }
}

/*!
 * \brief Dispatch the elements of an ETL container in a parallel manner and use an accumulator functor to accumulate the results.
 *
 * The functors will be called with slices of the original expression.
 *
 * \param expr The expression to slice
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param threshold The threshold for paralellization
 */
template <typename E1, typename E2, typename Functor>
inline void engine_dispatch_1d_slice_binary(E1&& expr1, E2&& expr2, Functor&& functor, size_t threshold) {
    using TT = value_t<E1>;

    static constexpr size_t S = default_intrinsic_traits<TT>::size;

    const size_t n = etl::size(expr1);

    if (n) {
        if (engine_select_parallel(n, threshold)) {
            const size_t T = std::min(n, etl::threads);

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                if constexpr (decay_traits<E1>::is_aligned && decay_traits<E2>::is_aligned && S > 1) {
                    if (n >= T * S) {
                        // In case there is enough data, we align it

                        const size_t n_aligned         = (n + (S - 1)) & ~(S - 1);
                        const size_t blocks            = n_aligned / S;
                        const size_t blocks_per_thread = blocks / T;
                        const size_t batch             = blocks_per_thread * S;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(functor, memory_slice<aligned>(expr1, t * batch, (t + 1) * batch),
                                                    memory_slice<aligned>(expr2, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(functor, memory_slice<aligned>(expr1, (T - 1) * batch, n), memory_slice<aligned>(expr2, (T - 1) * batch, n));
                    } else {
                        // Not enough data to consider aligning

                        const size_t batch = n / T;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(functor, memory_slice<unaligned>(expr1, t * batch, (t + 1) * batch),
                                                    memory_slice<unaligned>(expr2, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(functor, memory_slice<unaligned>(expr1, (T - 1) * batch, n),
                                                memory_slice<unaligned>(expr2, (T - 1) * batch, n));
                    }
                } else {
                    // If the data is not aligned in the first, don't make any effort to align it

                    const size_t batch = n / T;

                    for (size_t t = 0; t < T - 1; ++t) {
                        thread_engine::schedule(functor, memory_slice<unaligned>(expr1, t * batch, (t + 1) * batch),
                                                memory_slice<unaligned>(expr2, t * batch, (t + 1) * batch));
                    }

                    thread_engine::schedule(functor, memory_slice<unaligned>(expr1, (T - 1) * batch, n), memory_slice<unaligned>(expr2, (T - 1) * batch, n));
                }

                thread_engine::wait();
            }
        } else {
            functor(expr1, expr2);
        }
    }
}

/*!
 * \brief Dispatch the elements of an ETL container in a parallel manner and use an accumulator functor to accumulate the results.
 *
 * The functors will be called with slices of the original expression.
 *
 * \param expr The expression to slice
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param threshold The threshold for paralellization
 */
template <typename E, typename Functor, typename AccFunctor>
inline void engine_dispatch_1d_acc_slice(E&& expr, Functor&& functor, AccFunctor&& acc_functor, size_t threshold) {
    using TT = value_t<E>;

    static constexpr size_t S = default_intrinsic_traits<TT>::size;

    const size_t n = etl::size(expr);

    if (n) {
        if (engine_select_parallel(n, threshold)) {
            const size_t T = std::min(n, etl::threads);

            std::vector<TT> futures(T);

            auto sub_functor = [&futures, &functor](size_t t, auto&& sub_expr) { futures[t] = functor(sub_expr); };

            ETL_PARALLEL_SESSION {
                thread_engine::acquire();

                if constexpr (decay_traits<E>::is_aligned && S > 1) {
                    if (n >= T * S) {
                        // In case there is enough data, we align it

                        const size_t n_aligned         = (n + (S - 1)) & ~(S - 1);
                        const size_t blocks            = n_aligned / S;
                        const size_t blocks_per_thread = blocks / T;
                        const size_t batch             = blocks_per_thread * S;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(sub_functor, t, memory_slice<aligned>(expr, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(sub_functor, T - 1, memory_slice<aligned>(expr, (T - 1) * batch, n));
                    } else {
                        // Not enough data to consider aligning

                        const size_t batch = n / T;

                        for (size_t t = 0; t < T - 1; ++t) {
                            thread_engine::schedule(sub_functor, t, memory_slice<unaligned>(expr, t * batch, (t + 1) * batch));
                        }

                        thread_engine::schedule(sub_functor, T - 1, memory_slice<unaligned>(expr, (T - 1) * batch, n));
                    }
                } else {
                    // If the data is not aligned in the first, don't make any effort to align it

                    const size_t batch = n / T;

                    for (size_t t = 0; t < T - 1; ++t) {
                        thread_engine::schedule(sub_functor, t, memory_slice<unaligned>(expr, t * batch, (t + 1) * batch));
                    }

                    thread_engine::schedule(sub_functor, T - 1, memory_slice<unaligned>(expr, (T - 1) * batch, n));
                }

                thread_engine::wait();
            }

            // Accumulate the results
            for (auto fut : futures) {
                acc_functor(fut);
            }
        } else {
            acc_functor(functor(expr));
        }
    }
}

#else

/*!
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param n The size of the evaluation
 * \param threshold The parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool engine_select_parallel([[maybe_unused]] size_t n, [[maybe_unused]] size_t threshold = parallel_threshold) {
    return false;
}

/*!
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param select The secondary parallel selection
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool engine_select_parallel([[maybe_unused]] bool select) {
    return false;
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
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, [[maybe_unused]] size_t threshold) {
    cpp_assert(last >= first, "Range must be valid");

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
inline void engine_dispatch_1d_serial(Functor&& functor, size_t first, size_t last, size_t threshold) {
    engine_dispatch_1d(functor, first, last, threshold);
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
inline void engine_dispatch_1d_serial_cpu(Functor&& functor, size_t first, size_t last, size_t threshold) {
    engine_dispatch_1d(functor, first, last, threshold);
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
inline void engine_dispatch_1d_cpu(Functor&& functor, size_t first, size_t last, size_t threshold) {
    engine_dispatch_1d(functor, first, last, threshold);
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
 */
template <typename Functor>
inline void engine_dispatch_1d(Functor&& functor, size_t first, size_t last, [[maybe_unused]] bool select) {
    cpp_assert(last >= first, "Range must be valid");

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
 */
template <typename Functor>
inline void engine_dispatch_1d_serial(Functor&& functor, size_t first, size_t last, bool select) {
    engine_dispatch_1d(functor, first, last, select);
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
 */
template <typename Functor>
inline void engine_dispatch_1d_serial_cpu(Functor&& functor, size_t first, size_t last, bool select) {
    engine_dispatch_1d(functor, first, last, select);
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
 */
template <typename Functor>
inline void engine_dispatch_1d_cpu(Functor&& functor, size_t first, size_t last, bool select) {
    engine_dispatch_1d(functor, first, last, select);
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner and use an accumulator functor to accumulate the results
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param first The beginning of the range
 * \param last The end of the range
 * \param threshold The threshold for paralellization
 */
template <typename T, typename Functor, typename AccFunctor>
inline void engine_dispatch_1d_acc(Functor&& functor, AccFunctor&& acc_functor, size_t first, size_t last, [[maybe_unused]] size_t threshold) {
    cpp_assert(last >= first, "Range must be valid");

    const size_t n = last - first;

    if (n) {
        acc_functor(functor(first, last));
    }
}

/*!
 * \brief Dispatch the elements of an ETL container in a parallel manner and use an accumulator functor to accumulate the results.
 *
 * The functors will be called with slices of the original expression.
 *
 * \param expr The expression to slice
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param threshold The threshold for paralellization
 */
template <typename E, typename Functor, typename AccFunctor>
inline void engine_dispatch_1d_acc_slice(E&& expr, Functor&& functor, AccFunctor&& acc_functor, [[maybe_unused]] size_t threshold) {
    acc_functor(functor(expr));
}

#endif

} //end of namespace etl
