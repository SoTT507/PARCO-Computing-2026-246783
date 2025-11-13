#include "s_matrix.hpp"
#include "benchmark.hpp"
void* spmv_pthread_worker(void* arg) {
    PthreadArgs* args = static_cast<PthreadArgs*>(arg);
    const CSRMatrix* csr = args->csr;
    const double* x = args->x;
    double* y = args->y;
    
    for (int i = args->start_row; i < args->end_row; i++) {
        double sum = 0.0;
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
            sum += csr->values[j] * x[csr->col_idx[j]];
        }
        y[i] = sum;
    }
    
    return nullptr;
}

BenchmarkResult SparseMatrixBenchmark::benchmarkCSRPthreads(
    const CSRMatrix& csr, const std::vector<double>& x, int num_threads, int runs) {
    
    std::vector<double> times;
    
    for (int run = 0; run < runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> y(csr.rows, 0.0);
        std::vector<pthread_t> threads(num_threads);
        std::vector<PthreadArgs> thread_args(num_threads);
        
        int rows_per_thread = csr.rows / num_threads;
        int extra_rows = csr.rows % num_threads;
        
        // Create threads
        int current_row = 0;
        for (int i = 0; i < num_threads; i++) {
            int rows_this_thread = rows_per_thread + (i < extra_rows ? 1 : 0);
            
            thread_args[i] = {
                &csr, x.data(), y.data(), 
                current_row, current_row + rows_this_thread, i, nullptr
            };
            
            pthread_create(&threads[i], nullptr, spmv_pthread_worker, &thread_args[i]);
            current_row += rows_this_thread;
        }
        
        // Join threads
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], nullptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(duration);
    }
    
    BenchmarkResult result;
    result.calculate(times);
    return result;
}
