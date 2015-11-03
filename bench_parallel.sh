make clean
rm -rf results/*.cpm

unset ETL_PARALLEL
make -j9 release_debug/bin/benchmark && ./release_debug/bin/benchmark -c serial -t now

rm -rf release_debug
export ETL_PARALLEL=true
make -j9 release_debug/bin/benchmark && ./release_debug/bin/benchmark -c parallel -t now

mkdir reports
../cpm/release_debug/bin/cpm results
