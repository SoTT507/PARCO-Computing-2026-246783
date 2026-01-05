#!/bin/bash

module load GCCcore/13.2.0
module load zlib/1.2.13-GCCcore-13.2.0
module load binutils/2.40-GCCcore-13.2.0
module load GCC/13.2.0
module load CMake/3.27.6-GCCcore-13.2.0

# Sparse Matrix Benchmark Build Script
echo "=========================================="
echo "  Sparse Matrix Benchmark Build Script"
echo "=========================================="

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi
cd build

#check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake could not be found. Please install CMake 3.14 or later version"
    exit 1
fi

cmake --fresh ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

cmake --build .
if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi

mv ./d2_SpMV ../

cd ..

# cd thirdparty
# exec ./getmatrices.sh
# cd ..

echo "====================================================="
echo "           Build completed successfully :)           "
echo ""
echo " to run the benchmark enter ./d1_SpMV                "
echo " or double click on it if you are using the GUI"

echo "====================================================="
