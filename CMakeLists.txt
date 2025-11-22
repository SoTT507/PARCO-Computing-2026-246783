cmake_minimum_required(VERSION 3.10)
project(d1_SpMV)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE PROG_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio/mmio.c"  # Just include it directly
)

find_package(OpenMP REQUIRED)

add_executable(d1_SpMV ${PROG_SRC})

target_include_directories(d1_SpMV PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio"
)

target_link_libraries(d1_SpMV OpenMP::OpenMP_CXX)
target_compile_options(d1_SpMV PRIVATE ${OpenMP_CXX_FLAGS})

# Force mmio.c to be compiled as C++ (if needed)
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio/mmio.c" 
    PROPERTIES LANGUAGE CXX
)
