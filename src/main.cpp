#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << " ============= INITIATING BENCHMARK ============= " << std::endl;
    }

    // Get OpenMP thread count
    int omp_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        omp_threads = omp_get_num_threads();
    }

    std::vector<std::string> matrices = {
        // "thirdparty/1138_bus/1138_bus.mtx"    
        "thirdparty/audikw_1/audikw_1.mtx",
        "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
        "thirdparty/Serena/Serena.mtx",
        "thirdparty/Freescale1/Freescale1.mtx",
        "thirdparty/ldoor/ldoor.mtx",
        "thirdparty/G3_circuit/G3_circuit.mtx",
        "thirdparty/Transport/Transport.mtx"
    };

    if (rank == 0) {
        std::cout << " ============= BENCHMARK: MPI + OMP GUIDED ============= " << std::endl;
        std::cout << " MPI Ranks: " << size << " | OMP Threads: " << omp_threads << std::endl;
    }

    std::string csv_file = "mpi_spmv_results.csv";
    if (rank == 0) {
        std::cout << "--> Initializing CSV: " << csv_file << std::endl;
        SparseMatrixBenchmark::writeMPIcsvHeader(csv_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (const auto &path : matrices) {
        std::filesystem::path fs_path(path);
        std::string matrix_name = fs_path.stem().string();

        if (rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Processing matrix: " << matrix_name << std::endl;
            std::cout << "========================================" << std::endl;
        }

        // ============================================================
        // STEP 1: Load matrix on rank 0
        // ============================================================
        COOMatrix global;

        if (rank == 0) {
            try {
                global.readMatrixMarket(path);
                std::cout << "  Matrix loaded: " << global.rows << " x " << global.cols 
                          << ", nnz = " << global.nnz << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error loading matrix " << path << ": " << e.what() << std::endl;
                continue;
            }
        }

        // Broadcast dimensions
        MPI_Bcast(&global.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate on non-root ranks
        if (rank != 0) {
            global.row_idx.resize(global.nnz);
            global.col_idx.resize(global.nnz);
            global.values.resize(global.nnz);
        }

        // Broadcast matrix data
        MPI_Bcast(global.row_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(global.col_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(global.values.data(), global.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        // ============================================================
        // STEP 2: Create input vector (replicated on all ranks)
        // ============================================================
        std::vector<double> x_global(global.cols, 1.0);

        // ============================================================
        // STEP 3: Test 1D Partitioning
        // ============================================================
        if (rank == 0) {
            std::cout << "\n--- Testing 1D Partitioning ---" << std::endl;
        }

        // Measure vector broadcast time for 1D
        double broadcast_time_1d = 0.0;
        
        auto bcast_start = std::chrono::high_resolution_clock::now();
        
        // For 1D: vector is already replicated, but in practice you'd do:
        // MPI_Bcast(x_global.data(), global.cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Since we initialize it on all ranks, broadcast time is 0 (or measure it separately)
        
        auto bcast_end = std::chrono::high_resolution_clock::now();
        broadcast_time_1d = std::chrono::duration<double, std::milli>(bcast_end - bcast_start).count();

        // Create distributed matrix
        DistributedMatrix A1(global, Partitioning::OneD);
        
        // Get memory usage
        size_t local_mem_bytes_1 = A1.getLocalMemoryUsage();
        size_t max_mem_bytes_1 = 0;
        MPI_Reduce(&local_mem_bytes_1, &max_mem_bytes_1, 1, 
                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        double mem_mb_1 = max_mem_bytes_1 / (1024.0 * 1024.0);

        // Run benchmark
        BenchmarkResult res1 = SparseMatrixBenchmark::benchmark_spmv(A1, x_global, 10);

        // Add broadcast time to communication time for 1D
        // (since the vector needs to be available on all ranks)
        res1.avg_comm_time += broadcast_time_1d;

        if (rank == 0) {
            std::cout << "  Results:" << std::endl;
            std::cout << "    90th percentile: " << res1.percentile_90 << " ms" << std::endl;
            std::cout << "    Avg time: " << res1.average << " ms" << std::endl;
            std::cout << "    Avg comm: " << res1.avg_comm_time << " ms" << std::endl;
            std::cout << "    Avg comp: " << res1.avg_comp_time << " ms" << std::endl;
            std::cout << "    Max memory: " << mem_mb_1 << " MB" << std::endl;
            
            SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D", 
                                                  size, omp_threads, global.nnz, 
                                                  mem_mb_1, res1);
        }

        // ============================================================
        // STEP 4: Test 2D Partitioning (if size > 1)
        // ============================================================
        if (size > 1) {
            if (rank == 0) {
                std::cout << "\n--- Testing 2D Partitioning ---" << std::endl;
            }

            // Create distributed matrix with 2D partitioning
            DistributedMatrix A2(global, Partitioning::TwoD);
            
            // Get memory usage
            size_t local_mem_bytes_2 = A2.getLocalMemoryUsage();
            size_t max_mem_bytes_2 = 0;
            MPI_Reduce(&local_mem_bytes_2, &max_mem_bytes_2, 1, 
                       MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double mem_mb_2 = max_mem_bytes_2 / (1024.0 * 1024.0);

            // Run benchmark (communication is measured inside SpMV for 2D)
            BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x_global, 10);

            if (rank == 0) {
                std::cout << "  Results:" << std::endl;
                std::cout << "    90th percentile: " << res2.percentile_90 << " ms" << std::endl;
                std::cout << "    Avg time: " << res2.average << " ms" << std::endl;
                std::cout << "    Avg comm: " << res2.avg_comm_time << " ms" << std::endl;
                std::cout << "    Avg comp: " << res2.avg_comp_time << " ms" << std::endl;
                std::cout << "    Max memory: " << mem_mb_2 << " MB" << std::endl;
                
                SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D", 
                                                      size, omp_threads, global.nnz, 
                                                      mem_mb_2, res2);
            }
        } else if (rank == 0) {
            std::cout << "  Skipping 2D partitioning (requires >1 MPI process)" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ============================================================
  // WEAK SCALING BENCHMARK
  // ============================================================
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
      std::cout << "\n ============= WEAK SCALING (Random Matrix) ============= " << std::endl;
  }

  // Configuration for Weak Scaling - problem size grows with processors
  // Keep work per processor constant: nnz per processor = base_nnz_per_proc
  int base_nnz_per_proc = 1000000;  // 1M non-zeros per processor as baseline
  
  // Calculate total nnz for weak scaling
  int total_nnz_weak = base_nnz_per_proc * size;
  
  // For square matrices: n = sqrt(nnz / density), assuming density = 0.001
  double density = 0.001;  // 0.1% density typical for sparse matrices
  int weak_dim = static_cast<int>(sqrt(total_nnz_weak / density));
  
  std::string ws_csv = "mpi_weak_scaling.csv";
  if (rank == 0) {
      std::cout << "  Weak Scaling Configuration:" << std::endl;
      std::cout << "    Processors: " << size << std::endl;
      std::cout << "    Base nnz per processor: " << base_nnz_per_proc << std::endl;
      std::cout << "    Total nnz: " << total_nnz_weak << std::endl;
      std::cout << "    Matrix dimension: " << weak_dim << "x" << weak_dim << std::endl;
      std::cout << "    Expected density: " << density * 100 << "%" << std::endl;
      SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);
  }

  COOMatrix weak_global;
  
  if (rank == 0) {
      std::cout << "  Generating Random Sparse Matrix..." << std::endl;
      try {
          // Generate random sparse matrix with specified dimension and target nnz
          weak_global.generateRandomSparse(weak_dim, density, total_nnz_weak);
          std::cout << "    Actual nnz generated: " << weak_global.nnz << std::endl;
          std::cout << "    Actual density: " 
                    << (100.0 * weak_global.nnz) / (weak_global.rows * weak_global.cols) 
                    << "%" << std::endl;
      } catch (const std::exception& e) {
          std::cerr << "Error generating weak scaling matrix: " << e.what() << std::endl;
          // Continue with other benchmarks even if weak scaling fails
          MPI_Barrier(MPI_COMM_WORLD);
          goto cleanup;
      }
  }

  // Broadcast dimensions
  MPI_Bcast(&weak_global.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&weak_global.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&weak_global.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
      weak_global.row_idx.resize(weak_global.nnz);
      weak_global.col_idx.resize(weak_global.nnz);
      weak_global.values.resize(weak_global.nnz);
  }

  // Broadcast matrix data
  MPI_Bcast(weak_global.row_idx.data(), weak_global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(weak_global.col_idx.data(), weak_global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(weak_global.values.data(), weak_global.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x_weak(weak_global.cols, 1.0);

  // Run 1D Weak Scaling
  if (rank == 0) {
      std::cout << "\n--- Weak Scaling: 1D Partitioning ---" << std::endl;
  }
  
  DistributedMatrix A_weak_1d(weak_global, Partitioning::OneD);
  
  // METRIC: Memory Footprint
  size_t local_mem_1d = A_weak_1d.getLocalMemoryUsage();
  size_t max_mem_1d = 0;
  MPI_Reduce(&local_mem_1d, &max_mem_1d, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  double max_mem_mb_1d = max_mem_1d / (1024.0 * 1024.0);

  BenchmarkResult res_weak_1d = SparseMatrixBenchmark::benchmark_spmv(A_weak_1d, x_weak, 10);

  if (rank == 0) {
      SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "1D", 
                                           size, omp_threads, weak_global.nnz, 
                                           max_mem_mb_1d, res_weak_1d);
      std::cout << "    Avg time: " << res_weak_1d.average << " ms" << std::endl;
      std::cout << "    Avg comm: " << res_weak_1d.avg_comm_time << " ms" << std::endl;
      std::cout << "    Avg comp: " << res_weak_1d.avg_comp_time << " ms" << std::endl;
      std::cout << "    Max memory per rank: " << max_mem_mb_1d << " MB" << std::endl;
  }

  // Run 2D Weak Scaling if size > 1
  if (size > 1) {
      if (rank == 0) {
          std::cout << "\n--- Weak Scaling: 2D Partitioning ---" << std::endl;
      }
      
      DistributedMatrix A_weak_2d(weak_global, Partitioning::TwoD);
      
      size_t local_mem_2d = A_weak_2d.getLocalMemoryUsage();
      size_t max_mem_2d = 0;
      MPI_Reduce(&local_mem_2d, &max_mem_2d, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
      double max_mem_mb_2d = max_mem_2d / (1024.0 * 1024.0);
      
      BenchmarkResult res_weak_2d = SparseMatrixBenchmark::benchmark_spmv(A_weak_2d, x_weak, 10);
      
      if (rank == 0) {
          SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "2D", 
                                               size, omp_threads, weak_global.nnz, 
                                               max_mem_mb_2d, res_weak_2d);
          std::cout << "    Avg time: " << res_weak_2d.average << " ms" << std::endl;
          std::cout << "    Avg comm: " << res_weak_2d.avg_comm_time << " ms" << std::endl;
          std::cout << "    Avg comp: " << res_weak_2d.avg_comp_time << " ms" << std::endl;
          std::cout << "    Max memory per rank: " << max_mem_mb_2d << " MB" << std::endl;
      }
  }

  if (rank == 0) {
      std::cout << "\nWeak scaling results saved to: " << ws_csv << std::endl;
  }

  if (rank == 0) {
    std::cout << "\n ============= BENCHMARK COMPLETE ============= " << std::endl;
  }

  MPI_Finalize();
  return 0;
}
