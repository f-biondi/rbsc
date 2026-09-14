#!/bin/bash

mkdir -p src/rbsc/build
cd src/rbsc/build
cmake ..
cmake --build . --target rbsc
cmake --build . --target rbsc_aggressive
cd ../../../
mv src/rbsc/build/rbsc .
mv src/rbsc/build/rbsc_aggressive .

mkdir -p src/rsc/build
cd src/rsc/build
cmake ..
cmake --build . --target rsc
cd ../../../
mv src/rsc/build/rsc .

cd src/part_ref
g++ -std=c++11 -W -Wall -pedantic -O2 MDPmin.cc -o part_ref
cd ../../
mv src/part_ref/part_ref .

cd webgraph-rs
cargo build --release
cd ..
mv webgraph-rs/target/release/webgraph .
