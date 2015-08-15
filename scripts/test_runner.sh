#!/bin/bash
set -e

# Disable default options
export ETL_NO_DEFAULT=true
unset ETL_DEFAULTS
unset ETL_MKL
unset ETL_BLAS

# Use gcc
export CXX=g++-4.9.2
export LD=g++-4.9.2

echo "Test 1. GCC (debug default)"

make clean
make -j9 debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_1.xml

echo "Test 2. GCC (debug vectorize avx)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL -mavx2 -mavx"

make clean
make -j9 debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_2.xml

echo "Test 3. GCC (debug vectorize sse)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL -msse3 -msse4"

make clean
make -j9 debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_3.xml

echo "Test 4. GCC (debug mkl)"

unset ETL_DEFAULTS
export ETL_MKL=true

make clean
make -j9 debug/bin/etl_test
./debug/bin/etl_test
gcovr -x -b -r . --object-directory=debug/test > coverage_4.xml
