#!/bin/sh
set -e

function etl_run {
    make clean
    time make $ETL_THREADS debug/bin/etl_test
    time ./debug/bin/etl_test
}

# Disable default options
export ETL_NO_DEFAULT=true
unset ETL_DEFAULTS
unset ETL_MKL
unset ETL_BLAS
unset ETL_CUBLAS
unset ETL_CUFFT
unset ETL_CUDNN
unset ETL_GPU
unset ETL_EGBLAS

# Use gcc
export CXX=$ETL_GPP
export LD=$ETL_GPP

echo "Tests are compiled using $CXX compiler:"
$CXX --version

echo "Test 1. GCC (debug default)"

export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS -DCPP_UTILS_ASSERT_EXCEPTION"

etl_run 1

echo "Test 2. GCC (debug vectorize avx)"

export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS -DETL_VECTORIZE_FULL -mavx"

etl_run 2

echo "Test 3. GCC (debug vectorize sse)"

export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS -DETL_VECTORIZE_FULL -msse3 -msse4"

etl_run 3

echo "Test 4. GCC (debug mkl)"

export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS"
export ETL_MKL=true

etl_run 4

echo "Test 5. GCC (debug parallel)"

unset ETL_MKL
export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS -DETL_PARALLEL"

etl_run 5

echo "Test 6. GCC (debug vectorize sse avx parallel)"

unset ETL_MKL
export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS -DETL_PARALLEL -DETL_VECTORIZE_FULL -msse3 -msse4 -mavx"

etl_run 6

if [ "$ETL_NO_GPU" == "" ]
then
    echo "Test 7. GCC (debug cublas cufft)"

    export ETL_DEFAULTS="-DETL_DEBUG_THRESHOLDS"
    unset ETL_MKL
    export ETL_CUBLAS=true
    export ETL_CUFFT=true
    export ETL_CUDNN=true

    etl_run 7
fi
