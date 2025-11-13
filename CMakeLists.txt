cmake_minimum_required(VERSION 3.10)
project(d1_SpMV)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PCH_HEADER_FILE "include/pch.h")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
file(GLOB_RECURSE PROG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

find_package(OpenMP REQUIRED)

add_executable(d1_SpMV ${PROG_SRC})

target_include_directories(d1_SpMV PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(d1_SpMV PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
target_precompile_headers(d1_SpMV PRIVATE ${PCH_HEADER_FILE})
target_link_libraries(d1_SpMV OpenMP::OpenMP_CXX)
target_compile_options(d1_SpMV PRIVATE ${OpenMP_CXX_FLAGS})

