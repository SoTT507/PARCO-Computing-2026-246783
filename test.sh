#!/bin/bash
set -e
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
./getmatrices_test.sh
cd ..

TOTAL_CORES=$(nproc)

for MPI_PROCS in 1 2 4 8
  do
    OMP_THREADS=$((TOTAL_CORES / MPI_PROCS))
    if [ $OMP_THREADS -lt 1 ]; then
        OMP_THREADS=1
    fi

    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores

    echo "MPI=$MPI_PROCS  OMP=$OMP_THREADS"
    mpiexec --oversubscribe -n $MPI_PROCS ./d2_SpMV
  
    RESULT_NAME="plots/result_MPI${MPI_PROCS}_OMP${OMP_THREADS}.csv"
    cp mpi_spmv_results.csv "$RESULT_NAME"
  done
cd plot

python3 mpi_plot.py

cd ..

echo "====================================================="
echo "   Program built and run completed successfully :)   "
echo ""
echo " the benchmark has been performed and the graphs are "
echo " available in ./plots/                               "

echo "====================================================="
