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
./getmatrices.sh
cd ..


CORES_PER_NODE=64
NODES=2

MPI_PER_NODE_LIST="1 2 4 8 16 32 64"


for MPI_PER_NODE in $MPI_PER_NODE_LIST
do
    MPI_PROCS=$((MPI_PER_NODE * NODES))

    # Calculate OMP threads
    OMP_THREADS=$((CORES_PER_NODE / MPI_PER_NODE))

    # Safety check - ensure at least 1 thread
    if [ $OMP_THREADS -lt 1 ]; then
        OMP_THREADS=1
    fi

    # check that available cores per MPI process do not exceed
    if [ $OMP_THREADS -gt $((CORES_PER_NODE / MPI_PER_NODE)) ]; then
        OMP_THREADS=$((CORES_PER_NODE / MPI_PER_NODE))
    fi
    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PROC_BIND=spread
    # export OMP_PLACES=cores

    echo "MPI=$MPI_PROCS  OMP=$OMP_THREADS"

    mpiexec --oversubscribe -n $MPI_PROCS ./d2_SpMV --parallel-io

    RESULT_NAME="plots/result_MPI${MPI_PROCS}_OMP${OMP_THREADS}.csv"
    WRESULT_NAME="plots/weak_scaling_MPI${MPI_PROCS}_OMP${OMP_THREADS}.csv"
    if [ -f mpi_spmv_results.csv ]; then
        cp mpi_spmv_results.csv "$RESULT_NAME"
    fi
    if [ -f mpi_weak_scaling.csv ]; then
        cp mpi_weak_scaling.csv "$WRESULT_NAME"
    fi
  done

echo "====================================================="
echo "   Program built and run completed successfully :)   "
echo ""
echo " the benchmark has been performed and the graphs are "
echo " available in ./plots/                               "

echo "====================================================="
