#include "s_matrix.hpp"
#include "benchmark.hpp"
#include <iostream>

int main() {
    SparseMatrixBenchmark benchmark;
    // Add matrices with different sparsity patterns
    benchmark.addMatrixFile("thirdparty/bcsstk36/bcsstk36.mtx"); //~23k rows
    benchmark.addMatrixFile("thirdparty/F1/F1.mtx");//~34k rows
    benchmark.addMatrixFile("thirdparty/rdb968/rdb968.mtx");//~5k --> LOW sparsity
    benchmark.addMatrixFile("thirdparty/bcsstk25/bcsstk25.mtx");
    benchmark.addMatrixFile("thirdparty/af23560/af23560.mtx");
    //
    //
    //Considering a piece of code to run warmup directly from program
    //instead of running the program multiple times as of now
    //
    for(int i = 0; i<10; i++){
      benchmark.warmup(); //same as runFullBenchmark but does not save nor print results
      std::cout<< "Warmup run: " << i << std::endl;
    }

    // Run full benchmark
    benchmark.runFullBenchmark();

    return 0;
}
