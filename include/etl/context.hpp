//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Wrapper used in the context to force an implementation to be used
 */
template<typename T>
struct forced_impl {
    T impl;              ///< The impl to be used (if forced == true)
    bool forced = false; ///< Indicate if forced or default
};

/*!
 * \brief The contextual configuration of ETL
 */
struct context {
    bool serial = false;   ///< Force serial execution
    bool parallel = false; ///< Force parallel execution

    forced_impl<scalar_impl> scalar_selector;       ///< Force selector for scalar operations
    forced_impl<sum_impl> sum_selector;             ///< Forced selector for sum
    forced_impl<transpose_impl> transpose_selector; ///< Forced selector for transpose
    forced_impl<dot_impl> dot_selector;             ///< Forced selector for dot
    forced_impl<conv_impl> conv_selector;           ///< Forced selector for conv
    forced_impl<gemm_impl> gemm_selector;           ///< Forced selector for gemm
    forced_impl<outer_impl> outer_selector;         ///< Forced selector for outer product
    forced_impl<fft_impl> fft_selector;             ///< Forced selector for fft
};

/*!
 * \brief Return the configuration context of the current thread.
 * \return the configuration context of the current thread
 */
inline context& local_context(){
    static thread_local context local_context;
    return local_context;
}

namespace detail {

template<typename T>
forced_impl<T>& get_forced_impl();

template<>
inline forced_impl<scalar_impl>& get_forced_impl(){
    return local_context().scalar_selector;
}

template<>
inline forced_impl<sum_impl>& get_forced_impl(){
    return local_context().sum_selector;
}

template<>
inline forced_impl<transpose_impl>& get_forced_impl(){
    return local_context().transpose_selector;
}

template<>
inline forced_impl<dot_impl>& get_forced_impl(){
    return local_context().dot_selector;
}

template<>
inline forced_impl<conv_impl>& get_forced_impl(){
    return local_context().conv_selector;
}

template<>
inline forced_impl<gemm_impl>& get_forced_impl(){
    return local_context().gemm_selector;
}

template<>
inline forced_impl<outer_impl>& get_forced_impl(){
    return local_context().outer_selector;
}

template<>
inline forced_impl<fft_impl>& get_forced_impl(){
    return local_context().fft_selector;
}

/*!
 * \brief RAII helper for setting the context to serial
 */
struct serial_context {
    bool old_serial;

    serial_context(){
        old_serial = etl::local_context().serial;
        etl::local_context().serial = true;
    }

    ~serial_context(){
        etl::local_context().serial = old_serial;
    }

    operator bool(){
        return true;
    }
};

/*!
 * \brief RAII helper for setting the context to parallel
 */
struct parallel_context {
    bool old_parallel;

    parallel_context(){
        old_parallel = etl::local_context().parallel;
        etl::local_context().parallel = true;
    }

    ~parallel_context(){
        etl::local_context().parallel = old_parallel;
    }

    operator bool(){
        return true;
    }
};

/*!
 * \brief RAII helper for setting the context to a selected
 * implementation
 */
template<typename Selector, Selector V>
struct selected_context {
    forced_impl<Selector> old_selector;

    selected_context(){
        decltype(auto) selector = get_forced_impl<Selector>();

        old_selector = selector;

        selector.impl = V;
        selector.forced = true;
    }

    ~selected_context(){
        get_forced_impl<Selector>() = old_selector;
    }

    operator bool(){
        return true;
    }
};

} //end of namespace detail

/*!
 * \brief Define the start of an ETL serial section
 */
#define SERIAL_SECTION if(auto etl_serial_context__ = etl::detail::serial_context())

/*!
 * \brief Define the start of an ETL parallel section
 */
#define PARALLEL_SECTION if(auto etl_parallel_context__ = etl::detail::parallel_context())

/*!
 * \brief Define the start of an ETL selected section
 */
#define SELECTED_SECTION(v) if(auto etl_selected_context__ = etl::detail::selected_context<decltype(v), v>())

} //end of namespace etl
