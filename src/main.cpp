#include "s_matrix.hpp"
#include "benchmark.hpp"
#include <iostream>

int main() {
    SparseMatrixBenchmark benchmark;
    // Add matrices with different sparsity patterns
    


    benchmark.addMatrixFile("thirdparty/F1/F1.mtx");                 //~??k rows (depends on F1)
    benchmark.addMatrixFile("thirdparty/circuit5M/circuit5M.mtx");   //~5M rows (large)
    benchmark.addMatrixFile("thirdparty/kron_g500-logn21/kron_g500-logn21.mtx"); //~2^21 ≈ 2M rows
    benchmark.addMatrixFile("thirdparty/kron_g500-logn19/kron_g500-logn19.mtx"); //~2^19 ≈ 524k rows
    benchmark.addMatrixFile("thirdparty/audikw_1/audikw_1.mtx");     //~944k rows

    // benchmark.addMatrixFile("thirdparty/bcsstk36/bcsstk36.mtx"); //~23k rows
    // benchmark.addMatrixFile("thirdparty/bcsstk30/bcsstk30.mtx");//~34k rows
    // benchmark.addMatrixFile("thirdparty/rdb968/rdb968.mtx");//~5k --> LOW sparsity
    // benchmark.addMatrixFile("thirdparty/bcsstk25/bcsstk25.mtx");
    // benchmark.addMatrixFile("thirdparty/af23560/af23560.mtx");
    //
    //
    //Considering a piece of code to run warmup directly from program
    //instead of running the program multiple times as of now
    //
    for(int i = 1; i<=5; i++){
      benchmark.warmup(); //same as runFullBenchmark but does not save nor print results
      std::cout<< "Warmup run: " << i << std::endl;
    }

    // Run full benchmark
    benchmark.runFullBenchmark();

    return 0;
}
