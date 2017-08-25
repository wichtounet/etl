#!/bin/sh
set -e

function etl_run {
    make clean
    time make $ETL_THREADS debug/bin/etl_test

    echo "Run the tests"
    time ./debug/bin/etl_test --reporter=junit --out catch_report_${1}.xml

if [ "$ETL_LCOV" == "" ]
then
    gcovr -x -b -r . --object-directory=debug/test > coverage_${1}_raw.xml
    cov_clean coverage_${1}_raw.xml coverage_${1}.xml
else
    lcov --directory debug --rc lcov_branch_coverage=1 --capture --output-file coverage_${1}.dat
    lcov_cobertura.py -b debug -o coverage_${1}.xml coverage_${1}.dat
    sed -i 's/filename="..\//filename="/' coverage_${1}.xml
fi
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

# Enable coverage
export ETL_COVERAGE=true

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

    echo "Merge the coverage reports"

    if [ "$ETL_LCOV_MERGE" == "" ]
    then
        merge-xml-coverage.py -o coverage_report.xml coverage_1.xml coverage_2.xml coverage_3.xml coverage_4.xml coverage_5.xml coverage_6.xml coverage_7.xml
    else
        lcov --rc lcov_branch_coverage=1 -a coverage_1.dat -a coverage_2.dat -a coverage_3.dat -a coverage_4.dat -a coverage_5.dat -a coverage_6.dat -a coverage_7.dat -o coverage_full.dat
        lcov_cobertura.py -b debug -o coverage_report.xml coverage_full.dat
        sed -i 's/filename="..\//filename="/' coverage_report.xml
    fi
else
    echo "Merge the coverage reports"

    if [ "$ETL_LCOV_MERGE" == "" ]
    then
        merge-xml-coverage.py -o coverage_report.xml coverage_1.xml coverage_2.xml coverage_3.xml coverage_4.xml coverage_5.xml coverage_6.xml
    else
        lcov --rc lcov_branch_coverage=1 -a coverage_1.dat -a coverage_2.dat -a coverage_3.dat -a coverage_4.dat -a coverage_5.dat -a coverage_6.dat -o coverage_full.dat
        lcov_cobertura.py -b debug -o coverage_report.xml coverage_full.dat
        sed -i 's/filename="..\//filename="/' coverage_report.xml
    fi
fi

cp catch_report_1.xml catch_report.xml
