#!/bin/bash

module load GCCcore/12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load binutils/2.39-GCCcore-12.2.0
module load GCC/12.2.0
module load ncurses/6.3-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load OpenSSL/1.1
module load cURL/7.86.0-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0
module load libarchive/3.6.1-GCCcore-12.2.0
module load CMake/3.24.3-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
module load libxml2/2.10.3-GCCcore-12.2.0
module load libpciaccess/0.17-GCCcore-12.2.0
module load hwloc/2.8.0-GCCcore-12.2.0
module load libevent/2.1.12-GCCcore-12.2.0
module load UCX/1.13.1-GCCcore-12.2.0
module load libfabric/1.16.1-GCCcore-12.2.0
module load PMIx/4.2.2-GCCcore-12.2.0
module load UCC/1.1.0-GCCcore-12.2.0
module load OpenMPI/4.1.4-GCC-12.2.0

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

cd thirdparty
./getmatrices.sh
cd ..

echo "====================================================="
echo "           Build completed successfully :)           "
echo ""
echo " to run the benchmark enter ./d1_SpMV                "
echo " or double click on it if you are using the GUI"

echo "====================================================="
