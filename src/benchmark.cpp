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

void SparseMatrixBenchmark::writeBenchmarkHeader(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "matrix,format,threads,schedule,percentile_90,average,min_time,max_time,speedup,efficiency\n";
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
             << result.percentile_90 << ","
             << result.average << ","
             << result.min_time << ","
             << result.max_time << ","
             << speedup << ","
             << efficiency << "\n";
    }
}

void SparseMatrixBenchmark::writeDetailedResults(const std::string& matrix_name, const COOMatrix& coo, const CSRMatrix& csr) {
    std::string base_name = matrix_name;
    /* ====== IMPORTANT ======= */
    // Remove path and extension for clean filename
    size_t last_slash = base_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        base_name = base_name.substr(last_slash + 1);
    }
    size_t last_dot = base_name.find_last_of(".");
    if (last_dot != std::string::npos) {
        base_name = base_name.substr(0, last_dot);
    }

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    std::string csv_file = output_dir + "/" + base_name + "_results.csv";
    writeBenchmarkHeader(csv_file);

    std::vector<double> x = generateOnesVector(csr.cols);

    // Sequential benchmarks
    auto coo_seq = benchmarkCOOSequential(coo, x);
    auto csr_seq = benchmarkCSRSequential(csr, x);
    double seq_speedup = coo_seq.percentile_90 / csr_seq.percentile_90;

    writeBenchmarkResult(csv_file, base_name, "COO", 1, "sequential", coo_seq);
    writeBenchmarkResult(csv_file, base_name, "CSR", 1, "sequential", csr_seq, seq_speedup, 100.0);

    // Parallel benchmarks
    for (int threads : thread_counts) {
        auto omp_static = benchmarkCSROMPStatic(csr, x, threads);
        auto omp_dynamic = benchmarkCSROMPDynamic(csr, x, threads);
        auto omp_guided = benchmarkCSROMPGuided(csr, x, threads);
        auto pthreads = benchmarkCSRPthreads(csr, x, threads);

        double static_speedup = csr_seq.percentile_90 / omp_static.percentile_90;
        double static_efficiency = (static_speedup / threads) * 100.0;

        double dynamic_speedup = csr_seq.percentile_90 / omp_dynamic.percentile_90;
        double dynamic_efficiency = (dynamic_speedup / threads) * 100.0;

        double guided_speedup = csr_seq.percentile_90 / omp_guided.percentile_90;
        double guided_efficiency = (guided_speedup / threads) * 100.0;

        double pthread_speedup = csr_seq.percentile_90 / pthreads.percentile_90;
        double pthread_efficiency = (pthread_speedup / threads) * 100.0;

        writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_static",
                           omp_static, static_speedup, static_efficiency);
        writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_dynamic",
                           omp_dynamic, dynamic_speedup, dynamic_efficiency);
        writeBenchmarkResult(csv_file, base_name, "CSR", threads, "omp_guided",
                           omp_guided, guided_speedup, guided_efficiency);
        writeBenchmarkResult(csv_file, base_name, "CSR", threads, "pthreads",
                           pthreads, pthread_speedup, pthread_efficiency);
    }

    std::cout << "✓ Results saved to: " << csv_file << "\n";
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

void SparseMatrixBenchmark::runFullBenchmark() {
    std::cout << "--> Sparse Matrix-Vector Multiplication Benchmark <--\n\n";

    // Create output directory
    std::filesystem::create_directories(output_dir);

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

                double speedup = csr_seq.percentile_90 / omp_static.percentile_90;

                printf("%-12d", threads);
                printf("%-15.3f", omp_static.percentile_90);
                printf("%-15.3f", omp_dynamic.percentile_90);
                printf("%-15.3f", omp_guided.percentile_90);
                printf("%-15.3f", pthreads.percentile_90);
                printf("%-15.3f\n", speedup);
            }
            std::cout << "\n";

            // Write detailed results to CSV
            writeDetailedResults(file, coo, csr);

        } catch (const std::exception& e) {
            std::cout << "Error processing " << file << ": " << e.what() << "\n\n";
        }
    }

    std::cout << "All benchmark results saved to: " << output_dir << "/\n";
    std::cout << "Use 'python3 ./plots/plot.py' to generate graphs.\n";
}
