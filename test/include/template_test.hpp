//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_DECL( name, description, T ) \
    template<typename T> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )(); \
    TEST_CASE( name, description )

#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SECTION( Tn ) \
        SECTION( #Tn ) \
        { \
            INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )<Tn>(); \
        }

#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_DEFN( T ) \
    template<typename T> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )()

#define TEMPLATE_TEST_CASE_2( name, description, T, T1, T2 ) \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_DECL( name, description, T ) \
    { \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_SECTION( T1 ) \
        INTERNAL_CATCH_TEMPLATE_TEST_CASE_SECTION( T2 ) \
    } \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_DEFN( T )
