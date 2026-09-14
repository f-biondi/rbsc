cmake_minimum_required(VERSION 3.16)

project(be-proj)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

add_executable(rsc rsc.cpp)
