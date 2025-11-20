#include "s_matrix.hpp"
#include "benchmark.hpp"
#include <iostream>

int main() {
    SparseMatrixBenchmark benchmark;
    // Add matrices with different sparsity patterns
    benchmark.addMatrixFile("thirdparty/1138_bus/1138_bus.mtx");
    benchmark.addMatrixFile("thirdparty/fl2010/fl2010.mtx");
    //benchmark.addMatrixFile("thirdparty/F1/F1.mtx");
    //benchmark.addMatrixFile("thirdparty/circuit5M/circuit5M.mtx");
    //
    //
    //Considering a piece of code to run warmup directly from program
    //instead of running the program multiple times as of now
    //
    //for(int i = 0; i<=5; i++){
    //  benchmark.warmup(); //same as runFullBenchmark but does not save nor print results
    //}

    // Run full benchmark
    benchmark.runFullBenchmark();

    return 0;
}
