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
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=default

 echo "Configuration 2. Clang (vectorize_impl)"

export ETL_DEFAULTS="-DETL_VECTORIZE_IMPL"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_impl

echo "Configuration 3. Clang (vectorize_expr)"

export ETL_DEFAULTS="-DETL_VECTORIZE_EXPR"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_expr

echo "Configuration 4. Clang (vectorize_full)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_full

echo "Configuration 5. Clang (vectorize_full blas_mode)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_BLAS=true

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="blas_mode+vectorize_full"

echo "Configuration 6. Clang (vectorize_full mkl_mode)"

unset ETL_BLAS
export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_MKL=true

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="mkl_mode+vectorize_full"

unset ETL_DEFAULTS
unset ETL_MKL

# Continue with gcc
export CXX=g++-4.9.2
export LD=g++-4.9.2

echo "Configuration 1. GCC (default)"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=default

echo "Configuration 2. GCC (vectorize_impl)"

export ETL_DEFAULTS="-DETL_VECTORIZE_IMPL"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_impl

echo "Configuration 3. GCC (vectorize_expr)"

export ETL_DEFAULTS="-DETL_VECTORIZE_EXPR"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_expr

echo "Configuration 4. GCC (vectorize_full)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration=vectorize_full

echo "Configuration 5. GCC (vectorize_full blas_mode)"

export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_BLAS=true

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="blas_mode+vectorize_full"

echo "Configuration 6. GCC (vectorize_full mkl_mode)"

unset ETL_BLAS
export ETL_DEFAULTS="-DETL_VECTORIZE_FULL"
export ETL_MKL=true

make clean
make release/bin/benchmark
time ./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD` --configuration="mkl_mode+vectorize_full"
