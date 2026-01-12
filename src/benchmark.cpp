#include "benchmark.hpp"
#include <omp.h>

SparseMatrixBenchmark::SparseMatrixBenchmark() {
    // Default thread counts
    thread_counts = {1, 2, 4, 8, 16, 32, 64, 128};
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


void BenchmarkResult::calculate(const std::vector<double>& times,
                                const std::vector<double>& comms,
                                const std::vector<double>& comps) {
    run_times = times;
    comm_times = comms;
    comp_times = comps;
    if (times.empty()) return;

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    average = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Calculate average breakdown if available
    if (!comms.empty()) {
        avg_comm_time = std::accumulate(comms.begin(), comms.end(), 0.0) / comms.size();
    }
    if (!comps.empty()) {
        avg_comp_time = std::accumulate(comps.begin(), comps.end(), 0.0) / comps.size();
    }

    min_time = *std::min_element(times.begin(), times.end());
    max_time = *std::max_element(times.begin(), times.end());

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

//=============================================================
//===================== D2 Implementation =====================
//=============================================================

BenchmarkResult SparseMatrixBenchmark::benchmark_spmv(const DistributedMatrix& A,
                                       const std::vector<double>& x,
                                       int runs) {
    std::vector<double> y;

    std::vector<double> global_run_times;
    std::vector<double> global_comm_times;  // [ADDED]
    std::vector<double> global_comp_times;  // [ADDED]

    if (A.rank == 0) {
        global_run_times.reserve(runs);
        global_comm_times.reserve(runs);  // [ADDED]
        global_comp_times.reserve(runs);  // [ADDED]
    }

    for (int r = 0; r < runs; ++r) {
        // Synchronize
        MPI_Barrier(A.comm);
        auto t0 = std::chrono::high_resolution_clock::now();

        // Perform SpMV with timing breakdown
        double comm_time = 0.0, comp_time = 0.0;
        A.spmv(x, y, &comm_time, &comp_time);

        // MPI_Barrier(A.comm);
        auto t1 = std::chrono::high_resolution_clock::now();

        double local_dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double local_comm = comm_time;
        double local_comp = comp_time;

        double global_dt = 0.0, global_comm = 0.0, global_comp = 0.0;

        // Reduce MAX times for this run to Rank 0
        MPI_Reduce(&local_dt, &global_dt, 1, MPI_DOUBLE, MPI_MAX, 0, A.comm);
        MPI_Reduce(&local_comm, &global_comm, 1, MPI_DOUBLE, MPI_MAX, 0, A.comm);
        MPI_Reduce(&local_comp, &global_comp, 1, MPI_DOUBLE, MPI_MAX, 0, A.comm);

        if (A.rank == 0) {
            global_run_times.push_back(global_dt);
            global_comm_times.push_back(global_comm);  // [ADDED]
            global_comp_times.push_back(global_comp);  // [ADDED]
        }
    }

    BenchmarkResult result;

    // Calculate statistics on Rank 0
    if (A.rank == 0) {
        result.calculate(global_run_times, global_comm_times, global_comp_times);

        std::cout << std::left
            << std::setw(10) << "90th_perc"
            << std::setw(15) << "average"
            << std::setw(15) << "min_time"
            << std::setw(15) << "max_time"
            << std::setw(15) << "avg_comm"
            << std::setw(15) << "avg_comp"
            << "\n";

        std::cout << std::left
            << std::setw(10) << result.percentile_90
            << std::setw(15) << result.average
            << std::setw(15) << result.min_time
            << std::setw(15) << result.max_time
            << std::setw(15) << result.avg_comm_time
            << std::setw(15) << result.avg_comp_time
            << "\n";
    }

    // Broadcast results to all ranks
    double result_data[6] = {
        result.percentile_90, result.average, result.min_time,
        result.max_time, result.avg_comm_time, result.avg_comp_time
    };
    MPI_Bcast(result_data, 6, MPI_DOUBLE, 0, A.comm);

    result.percentile_90 = result_data[0];
    result.average = result_data[1];
    result.min_time = result_data[2];
    result.max_time = result_data[3];
    result.avg_comm_time = result_data[4];
    result.avg_comp_time = result_data[5];

    return result;
}
// =============================================================
// ================== NEW MPI CSV METHODS ======================
// =============================================================

void SparseMatrixBenchmark::writeMPIcsvHeader(const std::string& filename) {
// std::ios::trunc ensures we wipe the old file at the start of the job
    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if(file.is_open()){
        file << "matrix,partitioning,mpi_procs,omp_threads,nnz,max_mem_mb,"
            << "p90_ms,avg_ms,min_ms,max_ms,avg_comm_ms,avg_comp_ms,gflops\n"; // Added columns
    file.flush();
    file.close();
    } else {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
    }
}

void SparseMatrixBenchmark::writeMPIcsvRow(const std::string& filename,
                                              const std::string& matrix,
                                              const std::string& partitioning,
                                              int mpi_procs,
                                              int omp_threads,
                                              int nnz,
                                              double max_mem_mb, // [NEW ARGUMENT]
                                              const BenchmarkResult& r) {
    std::ofstream file(filename, std::ios::app);
    if(file.is_open()){
        double gflops = (r.average > 0) ? (2.0 * nnz / (r.average * 1e6)) : 0.0;

        file << matrix << ","
             << partitioning << ","
             << mpi_procs << ","
             << omp_threads << ","
             << nnz << ","
             << std::fixed << std::setprecision(4) << max_mem_mb << "," // [WRITE MEMORY]
             << std::setprecision(6) << r.percentile_90 << ","
             << r.average << ","
             << r.min_time << ","
             << r.max_time << ","
             << r.avg_comm_time << ","
             << r.avg_comp_time << ","
             << std::setprecision(4) << gflops << "\n";
        file.close();
    }
}
