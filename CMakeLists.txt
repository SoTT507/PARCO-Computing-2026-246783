cmake_minimum_required(VERSION 3.27)
project(d2_SpMV)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)      # Disable GNU extensions for portability
set(CMAKE_C_STANDARD 11)           # Set C standard for mmio.c
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE PROG_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio/mmio.c"  #Just Include mmio directly in d2_SpMV
)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(d2_SpMV ${PROG_SRC})

target_include_directories(d2_SpMV PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio"
    "${MPI_CXX_INCLUDE_DIRS}"
)

target_link_libraries(d2_SpMV 
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
)
target_compile_options(d2_SpMV PRIVATE ${OpenMP_CXX_FLAGS})

# Force mmio.c to be compiled as C++ (if needed)
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mmio/mmio.c" 
    PROPERTIES LANGUAGE CXX
)
