#include "s_matrix.hpp"
#include "benchmark.hpp"

int main() {
    SparseMatrixBenchmark benchmark;
    
    // Add matrices with different sparsity patterns
    benchmark.addMatrixFile("thirdparty/1138_bus/1138_bus.mtx");
    benchmark.addMatrixFile("thirdparty/fl2010/fl2010.mtx"); 
    benchmark.addMatrixFile("thirdparty/F1/F1.mtx"); 
    benchmark.addMatrixFile("thirdparty/circuit5M/circuit5M.mtx"); 
    
    // Or generate random matrices with different sparsity
    // std::cout << "Generating random matrices with different sparsity...\n";
    
    // std::vector<double> sparsities = {0.99, 0.95, 0.90, 0.80, 0.50};
    // for (double sparsity : sparsities) {
        // COOMatrix coo(5000, 5000);
        // coo.generateRandomSparse(5000, sparsity);
        // CSRMatrix csr(coo);
        //add benchmark for randomly generated sparse matrix
    // }
    
    // Run full benchmark
    benchmark.runFullBenchmark();
    
    return 0;
}
