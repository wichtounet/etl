#!/bin/bash
set -e

# Disable default options
export ETL_NO_DEFAULT=true
unset ETL_DEFAULTS
unset ETL_MKL
unset ETL_BLAS

# Use gcc
export CXX=$ETL_GPP
export LD=$ETL_GPP

echo "Test 1. GCC (debug default)"

export ETL_DEFAULTS="-DCPP_UTILS_ASSERT_EXCEPTION"

make clean
make $ETL_THREADS debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_1.xml

echo "Test 2. GCC (debug vectorize avx)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL -mavx"

make clean
make $ETL_THREADS debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_2.xml

echo "Test 3. GCC (debug vectorize sse)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL -msse3 -msse4"

make clean
make $ETL_THREADS debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_3.xml

echo "Test 4. GCC (debug mkl)"

unset ETL_DEFAULTS
export ETL_MKL=true

make clean
make $ETL_THREADS debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_4.xml

echo "Merge the coverage reports"

merger coverage_1.xml coverage_2.xml coverage_3.xml coverage_4.xml coverage_report.xml
