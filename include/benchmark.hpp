#include "pch.h"
#include "s_matrix.hpp"

class BenchmarkResult {
public:
    double percentile_90;
    double average;
    double min_time;
    double max_time;
    std::vector<double> run_times;
    
    void calculate(const std::vector<double>& times);
};

class SparseMatrixBenchmark {
private:
    std::vector<std::string> matrix_files;
    std::vector<int> thread_counts;
    
public:
    SparseMatrixBenchmark();
    
    void addMatrixFile(const std::string& filepath);
    void setThreadCounts(const std::vector<int>& counts);
    
    // Sequential implementations
    BenchmarkResult benchmarkCOOSequential(const COOMatrix& coo, const std::vector<double>& x, int runs = 10);
    BenchmarkResult benchmarkCSRSequential(const CSRMatrix& csr, const std::vector<double>& x, int runs = 10);
    
    // OpenMP parallel implementations
    BenchmarkResult benchmarkCSROMPStatic(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    BenchmarkResult benchmarkCSROMPDynamic(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    BenchmarkResult benchmarkCSROMPGuided(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    
    // Pthreads implementation
    BenchmarkResult benchmarkCSRPthreads(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    
    // Complete benchmarking
    void runFullBenchmark();
    
    // Utility functions
    static std::vector<double> generateRandomVector(int size);
    static std::vector<double> generateOnesVector(int size);
};

