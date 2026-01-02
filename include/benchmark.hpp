#include "pch.h"
#include "s_matrix.hpp"
#include "d_matrix.hpp"

struct MPITiming {
    double spmv_time_ms;
};

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
    std::string output_dir = "plots/";

public:
    SparseMatrixBenchmark();

    void addMatrixFile(const std::string& filepath);
    void setThreadCounts(const std::vector<int>& counts);
    void setOutputDirectory(const std::string& dir);

    // Benchmark methods
    BenchmarkResult benchmarkCOOSequential(const COOMatrix& coo, const std::vector<double>& x, int runs = 10);
    BenchmarkResult benchmarkCSRSequential(const CSRMatrix& csr, const std::vector<double>& x, int runs = 10);
    BenchmarkResult benchmarkCSROMPStatic(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    BenchmarkResult benchmarkCSROMPDynamic(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    BenchmarkResult benchmarkCSROMPGuided(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);
    BenchmarkResult benchmarkCSRPthreads(const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs = 10);

    // CSV writing methods
    void writeBenchmarkHeader(const std::string& filename);
    void writeBenchmarkResult(const std::string& filename, const std::string& matrix_name,
                            const std::string& format, int threads, const std::string& schedule,
                            const BenchmarkResult& result, double speedup = 0.0, double efficiency = 0.0);
    void warmup();

    // Utility methods
    static std::vector<double> generateRandomVector(int size);
    static std::vector<double> generateOnesVector(int size);

    // Main benchmark runner
    void runFullBenchmark();
   

  // ====================== D2 =======================
  // ================ IMPLEMENTATION =================
  // ===================== MPI =======================

    static MPITiming benchmark_spmv(const DistributedMatrix& A, const std::vector<double>& x, int runs);
    static void write_csv_header(const std::string& filename);
    static void write_csv_row(const std::string& filename, const std::string& matrix,
                              const std::string& partitioning, int mpi_procs,
                              int omp_threads, const MPITiming& t);
};
