cmake_minimum_required(VERSION 3.16)

project(be-cuda-proj LANGUAGES CXX CUDA)

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -g")
set(CMAKE_CUDA_FLAGS "-O3 -g")

find_package(OpenMP REQUIRED)

add_executable(rbsc rbsc.cu)
add_executable(rbsc_aggressive rbsc_aggressive.cu)

set_target_properties(rbsc PROPERTIES CUDA_ARCHITECTURES "90")
set_target_properties(rbsc_aggressive PROPERTIES CUDA_ARCHITECTURES "90")

target_compile_options(rbsc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_compile_options(rbsc_aggressive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)

target_link_libraries(rbsc PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(rbsc_aggressive PRIVATE OpenMP::OpenMP_CXX)
