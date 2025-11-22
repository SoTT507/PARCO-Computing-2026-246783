#include "benchmark.hpp"
#include <omp.h>

SparseMatrixBenchmark::SparseMatrixBenchmark() {
    // Default thread counts
    thread_counts = {1, 2, 4, 8, 16};
    // Create output directory
    std::filesystem::create_directories(output_dir);
}

void SparseMatrixBenchmark::addMatrixFile(const std::string& filepath) {
    matrix_files.push_back(filepath);
}

void SparseMatrixBenchmark::setThreadCounts(const std::vector<int>& counts) {
    thread_counts = counts;
}
void SparseMatrixBenchmark::setOutputDirectory(const std::string& dir) {
    output_dir = dir;
    std::filesystem::create_directories(output_dir);
}


void BenchmarkResult::calculate(const std::vector<double>& times) {
    run_times = times;
    if (times.empty()) return;

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    average = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    min_time = *std::min_element(times.begin(), times.end());
    max_time = *std::max_element(times.begin(), times.end());

    // 90th percentile - fixed index calculation
    size_t index = static_cast<size_t>(std::ceil(0.9 * times.size())) - 1;
    if (index >= sorted_times.size()) index = sorted_times.size() - 1;
    percentile_90 = sorted_times[index];
}


BenchmarkResult SparseMatrixBenchmark::benchmarkCOOSequential(
    const COOMatrix& coo, const std::vector<double>& x, int runs) {

    std::vector<double> y(coo.rows, 0.0);
    std::vector<double> times;
    times.reserve(runs);

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

    std::vector<double> y(csr.rows, 0.0);
    std::vector<double> times;
    times.reserve(runs);

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

    if (num_threads <= 0) num_threads = 1;
    
    std::vector<double> y(csr.rows, 0.0);
    std::vector<double> times;
    times.reserve(runs);

    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(num_threads)
        {
          #pragma omp for schedule(static)
          for (int i = 0; i < csr.rows; i++) {
              double sum = 0.0;
              for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                  sum += csr.values[j] * x[csr.col_idx[j]];
              }
              y[i] = sum;
          }
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

    if (num_threads <= 0) num_threads = 1;
    
    std::vector<double> y(csr.rows, 0.0);
    std::vector<double> times;
    times.reserve(runs);

    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        // OpenMP dynamic scheduling
        #pragma omp parallel num_threads(num_threads)
        {
          #pragma omp for schedule(dynamic) 
          for (int i = 0; i < csr.rows; i++) {
              double sum = 0.0;
              for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                  sum += csr.values[j] * x[csr.col_idx[j]];
              }
             y[i] = sum;
          }
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

    if (num_threads <= 0) num_threads = 1;
    
    std::vector<double> y(csr.rows, 0.0);
    std::vector<double> times;
    times.reserve(runs);

    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        
        #pragma omp parallel num_threads(num_threads)
        {

          #pragma omp for schedule(guided)
          for (int i = 0; i < csr.rows; i++) {
              double sum = 0.0;
              for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
                  sum += csr.values[j] * x[csr.col_idx[j]];
              }
              y[i] = sum;
          }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }

    BenchmarkResult result;
    result.calculate(times);
    return result;
}


void SparseMatrixBenchmark::writeBenchmarkHeader(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "matrix,format,threads,schedule,percentile_90,average,min_time,max_time,speedup,efficiency\n";
        file.close();
    } else {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
    }
}

void SparseMatrixBenchmark::writeBenchmarkResult(const std::string& filename, const std::string& matrix_name,
                        const std::string& format, int threads, const std::string& schedule,
                        const BenchmarkResult& result, double speedup, double efficiency) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << matrix_name << ","
             << format << ","
             << threads << ","
             << schedule << ","
             << std::fixed << std::setprecision(6) << result.percentile_90 << ","
             << result.average << ","
             << result.min_time << ","
             << result.max_time << ","
             << std::setprecision(3) << speedup << ","
             << std::setprecision(2) << efficiency << "\n";
        file.close();
    } else {
        std::cerr << "Error: Could not append to file: " << filename << std::endl;
    }
}

// ==================== Utility Functions ====================

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

void SparseMatrixBenchmark::runFullBenchmark() {
    std::cout << "--> Sparse Matrix-Vector Multiplication Benchmark <--\n\n";
    
    // Create output directory
    std::filesystem::create_directories(output_dir);

    for (const auto& file : matrix_files) {
        std::cout << "Testing matrix: " << file << "\n";
        std::cout << "=============================================\n";

        std::string base_name = std::filesystem::path(file).stem().string();
        
        std::string csv_file = output_dir + "/" + base_name + "_results.csv";
        writeBenchmarkHeader(csv_file);

        try {
            COOMatrix coo(file);
            CSRMatrix csr(coo);

            std::vector<double> x = generateOnesVector(csr.cols);

            // Sequential benchmarks
            auto coo_seq = benchmarkCOOSequential(coo, x);
            auto csr_seq = benchmarkCSRSequential(csr, x);
            
            // Avoid division by zero
            double seq_speedup = (csr_seq.percentile_90 > 0) ? 
                coo_seq.percentile_90 / csr_seq.percentile_90 : 0.0;

            writeBenchmarkResult(csv_file, base_name, "COO", 1, "sequential", coo_seq);
            writeBenchmarkResult(csv_file, base_name, "CSR", 1, "sequential", csr_seq, seq_speedup, 100.0);

            std::cout << "Sequential Results:\n";
            std::cout << "COO - 90th percentile: " << coo_seq.percentile_90 << " ms\n";
            std::cout << "CSR - 90th percentile: " << csr_seq.percentile_90 << " ms\n";
            std::cout << "Speedup CSR vs COO: " << seq_speedup << "x\n\n";

            // Parallel benchmarks
            std::cout << "Parallel Results (90th percentile in ms):\n\n";

            std::cout << std::left
                << std::setw(10) << "Threads"
                << std::setw(15) << "OMP Static"
                << std::setw(15) << "OMP Dynamic"
                << std::setw(15) << "OMP Guided"
                << "\n";

            for (int threads : thread_counts) {
                if (threads <= 0) continue;

                auto omp_static = benchmarkCSROMPStatic(csr, x, threads);
                auto omp_dynamic = benchmarkCSROMPDynamic(csr, x, threads);
                auto omp_guided = benchmarkCSROMPGuided(csr, x, threads);

                // Calculate speedup and efficiency with safety checks
                double static_speedup = (omp_static.percentile_90 > 0) ? 
                    csr_seq.percentile_90 / omp_static.percentile_90 : 0.0;
                double static_efficiency = (threads > 0) ? (static_speedup / threads) * 100.0 : 0.0;

                double dynamic_speedup = (omp_dynamic.percentile_90 > 0) ? 
                    csr_seq.percentile_90 / omp_dynamic.percentile_90 : 0.0;
                double dynamic_efficiency = (threads > 0) ? (dynamic_speedup / threads) * 100.0 : 0.0;

                double guided_speedup = (omp_guided.percentile_90 > 0) ? 
                    csr_seq.percentile_90 / omp_guided.percentile_90 : 0.0;
                double guided_efficiency = (threads > 0) ? (guided_speedup / threads) * 100.0 : 0.0;

                // Write results
                writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_static",
                                   omp_static, static_speedup, static_efficiency);
                writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_dynamic",
                                   omp_dynamic, dynamic_speedup, dynamic_efficiency);
                writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_guided",
                                   omp_guided, guided_speedup, guided_efficiency);

                std::cout << std::left
                        << std::setw(10) << threads
                        << std::setw(15) << std::fixed << std::setprecision(3) << omp_static.percentile_90
                        << std::setw(15) << omp_dynamic.percentile_90
                        << std::setw(15) << omp_guided.percentile_90
                        << "\n";
            }
            std::cout << "\n";

        } catch (const std::exception& e) {
            std::cout << "Error processing " << file << ": " << e.what() << "\n\n";
        }
    }

    std::cout << "All benchmark results saved to: " << output_dir << "/\n";
}

void SparseMatrixBenchmark::warmup() {
    for (const auto& file : matrix_files) {
        try {
            COOMatrix coo(file);
            CSRMatrix csr(coo);
            std::vector<double> x = generateOnesVector(csr.cols);

            // Run each benchmark once to warm up
            benchmarkCOOSequential(coo, x, 1);
            benchmarkCSRSequential(csr, x, 1);

            for (int threads : thread_counts) {
                if (threads <= 0) continue;
                benchmarkCSROMPStatic(csr, x, threads, 1);
                benchmarkCSROMPDynamic(csr, x, threads, 1);
                benchmarkCSROMPGuided(csr, x, threads, 1);
            }
        } catch (const std::exception& e) {
            std::cout << "Error in warmup for " << file << ": " << e.what() << "\n";
        }
    }
}
