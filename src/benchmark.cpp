#include "benchmark.hpp"


void BenchmarkResult::calculate(const std::vector<double>& times) {
    run_times = times;
    if (times.empty()) return;
    
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    average = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    min_time = *std::min_element(times.begin(), times.end());
    max_time = *std::max_element(times.begin(), times.end());
    
    // 90th percentile
    size_t index = static_cast<size_t>(std::ceil(0.9 * times.size())) - 1;
    percentile_90 = sorted_times[index];
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCOOSequential(
    const COOMatrix& coo, const std::vector<double>& x, int runs) {
    
    std::vector<double> y(coo.rows);
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // COO SpMV
        std::fill(y.begin(), y.end(), 0.0);
        for (int i = 0; i < coo.nnz; i++) {
            y[coo.row_idx[i]] += coo.values[i] * x[coo.col_idx[i]];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCSRSequential(
    const CSRMatrix& csr, const std::vector<double>& x, int runs) {
    
    std::vector<double> y(csr.rows);
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        csr.spmv(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCSROMPStatic(
    const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs) {
    
    std::vector<double> y(csr.rows);
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // OpenMP static scheduling
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < csr.rows; i++) {
            double sum = 0.0;
            for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                sum += csr.values[j] * x[csr.col_idx[j]];
            }
            y[i] = sum;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCSROMPDynamic(
    const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs) {
    
    std::vector<double> y(csr.rows);
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // OpenMP dynamic scheduling
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < csr.rows; i++) {
            double sum = 0.0;
            for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                sum += csr.values[j] * x[csr.col_idx[j]];
            }
            y[i] = sum;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCSROMPGuided(
    const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs) {
    
    std::vector<double> y(csr.rows);
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // OpenMP guided scheduling
        #pragma omp parallel for schedule(guided) num_threads(num_threads)
        for (int i = 0; i < csr.rows; i++) {
            double sum = 0.0;
            for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                sum += csr.values[j] * x[csr.col_idx[j]];
            }
            y[i] = sum;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}

//======================== UTIL and MAIN Benchmark =========================//


std::vector<double> SparseMatrixBenchmark::generateRandomVector(int size) {
    std::vector<double> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }
    return vec;
}

std::vector<double> SparseMatrixBenchmark::generateOnesVector(int size) {
    return std::vector<double>(size, 1.0);
}

SparseMatrixBenchmark::SparseMatrixBenchmark() {
    // Default thread counts
    thread_counts = {1, 2, 4, 8, 16, 32, 64};
}

void SparseMatrixBenchmark::addMatrixFile(const std::string& filepath) {
    matrix_files.push_back(filepath);
}

void SparseMatrixBenchmark::setThreadCounts(const std::vector<int>& counts) {
    thread_counts = counts;
}

void SparseMatrixBenchmark::runFullBenchmark() {
    std::cout << "--> Sparse Matrix-Vector Multiplication Benchmark <--\n\n";
    
    for (const auto& file : matrix_files) {
        std::cout << "Testing matrix: " << file << "\n";
        std::cout << "=============================================\n";
        
        try {
            COOMatrix coo(file);
            CSRMatrix csr(coo);
            
            std::vector<double> x = generateOnesVector(csr.cols);
            
            // Sequential benchmarks
            auto coo_seq = benchmarkCOOSequential(coo, x);
            auto csr_seq = benchmarkCSRSequential(csr, x);
            
            std::cout << "Sequential Results:\n";
            std::cout << "COO - 90th percentile: " << coo_seq.percentile_90 << " ms\n";
            std::cout << "CSR - 90th percentile: " << csr_seq.percentile_90 << " ms\n";
            std::cout << "Speedup CSR vs COO: " << coo_seq.percentile_90 / csr_seq.percentile_90 << "x\n\n";
            
            // Parallel benchmarks
            printf("Parallel Results (90th percentile in ms):\n");
            printf("%-12s", "Threads");
            printf("%-15s", "OMP Static");
            printf("%-15s", "OMP Dynamic");
            printf("%-15s", "OMP Guided");
            printf("%-15s", "Pthreads");
            printf("%-15s\n", "Speedup");
            
            for (int threads : thread_counts) {
                auto omp_static = benchmarkCSROMPStatic(csr, x, threads);
                auto omp_dynamic = benchmarkCSROMPDynamic(csr, x, threads);
                auto omp_guided = benchmarkCSROMPGuided(csr, x, threads);
                auto pthreads = benchmarkCSRPthreads(csr, x, threads);

                double perc_90
                double speedup = csr_seq.percentile_90 / omp_static.percentile_90;
                
                printf("%-12d", threads);
                printf("%-15.3f", omp_static.percentile_90);
                printf("%-15.3f", omp_dynamic.percentile_90);
                printf("%-15.3f", omp_guided.percentile_90);
                printf("%-15.3f", pthreads.percentile_90);
                printf("%-15.3f\n", speedup);
            }
            std::cout << "\n";
            
        } catch (const std::exception& e) {
            std::cout << "Error processing " << file << ": " << e.what() << "\n\n";
        }
    }
}
