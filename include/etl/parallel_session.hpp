//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief RAII helper for run and validating parallel session
 */
template<typename T>
struct parallel_session {
    /*!
     * \brief Default construct a parallel session
     *
     * This sets the parallel session as active and makes sure that no previous
     * parallel session was running.
     */
    parallel_session() {
        cpp_assert(!active, "Parallel session cannot be nested");

        active = true;
    }

    /*!
     * \brief Destruct a parallel session
     *
     * This disable the parallel session
     */
    ~parallel_session() {
        active = false;
    }

    /*!
     * \brief Does nothing, simple trick for macro to be nice
     */
    operator bool() {
        return true;
    }

    static bool active; ///< Indicates if the parallel session is active
};

template <typename T>
bool parallel_session<T>::active = false;

} //end of namespace detail

/*!
 * \brief Indicates if a parallel session is currently active
 * \return true if a parallel section is active, false otherwise
 */
inline bool is_parallel_session(){
    return detail::parallel_session<bool>::active;
}

/*!
 * \brief Define the start of an ETL parallel session
 */
#define ETL_PARALLEL_SESSION if (auto etl_parallel_session__ = etl::detail::parallel_session<bool>())

} //end of namespace etl
