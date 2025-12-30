#include "distributed_matrix.hpp"
#include <mpi.h>
#include "benchmark.hpp"
#include <iostream>

int main(int argc, char** argv) {
  //region: Deliverable 1 main code
  // SparseMatrixBenchmark benchmark;

    // benchmark.addMatrixFile("thirdparty/F1/F1.mtx");                 //~340k rows
    // benchmark.addMatrixFile("thirdparty/af_shell7/af_shell7.mtx");   //~5M rows
    // benchmark.addMatrixFile("thirdparty/mario002/mario002.mtx"); //~2M rows
    // benchmark.addMatrixFile("thirdparty/kron_g500-logn19/kron_g500-logn19.mtx"); //~524k rows
    // benchmark.addMatrixFile("thirdparty/msdoor/msdoor.mtx");     //~944k rows

    // benchmark.addMatrixFile("thirdparty/bcsstk36/bcsstk36.mtx"); //~23k rows
    // benchmark.addMatrixFile("thirdparty/bcsstk30/bcsstk30.mtx");//~34k rows
    // benchmark.addMatrixFile("thirdparty/rdb968/rdb968.mtx");//~5k --> LOW sparsity
    // benchmark.addMatrixFile("thirdparty/bcsstk25/bcsstk25.mtx");
    // benchmark.addMatrixFile("thirdparty/af23560/af23560.mtx");
    //
    //
    //Loop to run warmup directly from program
    //instead of running the program multiple times
    //
    // for(int i = 1; i<=5; i++){
      // benchmark.warmup(); //same as runFullBenchmark but does not save nor print results
      // std::cout<< "Warmup run: " << i << std::endl;
    // }

    // Run full benchmark
    // benchmark.runFullBenchmark();
  //endregion

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from rank " << rank << " of " << size << std::endl;

    //TODO MPI implementation
    
    MPI_Finalize();
    return 0;
}
