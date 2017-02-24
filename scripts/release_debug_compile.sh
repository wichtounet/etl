#!/bin/bash

make clean

max_memory="0.0"
total="0.0"

start=$(date "+%s.%N")

for f in test/src/*.cpp
do
    results=$(/usr/bin/time -v make release_debug/$f.o 2>&1)
    rss=$(echo "$results" | grep "Maximum resident set size" | rev | cut -d" " -f1 | rev)
    user_time=$(echo "$results" | grep "User time" | rev | cut -d" " -f1 | rev)
    elapsed=$(echo "$results" | grep "Elapsed" | rev | cut -d" " -f1 | rev)
    lines=$(cat $f | wc -l)
    memory=$(echo "scale=2; $rss/1024/4" | bc -l)
    relative=$(echo "1000.0 * ($user_time/$lines)" | bc -l)
    relative_fixed=$(echo "scale=2; $relative/1.0" | bc -l)
    echo "$f => $elapsed => ${memory}MB => ${relative_fixed} ms/l"

    if [ $(echo "$max_memory < $memory" | bc) -eq 1 ]
    then
        max_memory=$memory
    fi
done

end=$(date "+%s.%N")

runtime=$(echo "scale=3; ($end - $start) / 1.0" | bc -l)

echo "Max memory: $max_memory"
echo "Total time: $runtime"
