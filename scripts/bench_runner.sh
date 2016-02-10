#!/bin/bash
set -e

# Disable default options
export ETL_NO_DEFAULT=true
unset ETL_DEFAULTS
unset ETL_MKL
unset ETL_BLAS

# Start with clang
export CXX=clang++
export LD=clang++

echo "Configuration 1. Clang (default)"

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=default

echo "Configuration 2. Clang (vectorize_full)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_full

echo "Configuration 3. Clang (vectorize_full mkl_mode)"

unset ETL_BLAS
export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_MKL=true

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="mkl_mode+vectorize_full"

unset ETL_DEFAULTS
unset ETL_MKL

# Continue with gcc
export CXX=$ETL_GPP
export LD=$ETL_GPP

echo "Configuration 1. GCC (default)"

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=default

echo "Configuration 2. GCC (vectorize_full)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_full

echo "Configuration 3. GCC (vectorize_full mkl_mode)"

unset ETL_BLAS
export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_MKL=true

make clean
make -j6 release/bin/benchmark
time ./release/bin/benchmark $BENCH_ARGS --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="mkl_mode+vectorize_full"
