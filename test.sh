#!/bin/bash

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
mv ./libmmio.a ../

cd ..

cd thirdparty
exec ./getmatrices_test.sh
cd ..

export OMP_NUM_THREADS=4
echo "Running MPI+OpenMP SpMV benchmarks"
mpirun -np 4 ./d2_SpMV
mpirun -np 8 ./d2_SpMV
mpirun -np 16 ./d2_SpMV

cd plot

python3 mpi_plot.py

cd ..

echo "====================================================="
echo "   Program built and run completed successfully :)   "
echo ""
echo " the benchmark has been performed and the graphs are "
echo " available in ./plots/                               "

echo "====================================================="
