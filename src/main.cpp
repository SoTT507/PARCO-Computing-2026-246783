#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << " ============= INITIATING BENCHMARK ============= "
              << std::endl;
  }

  // Correct OpenMP thread count
  int omp_threads = 1;
  #pragma omp parallel
  {
  #pragma omp single
    omp_threads = omp_get_num_threads();
  }

  std::vector<std::string> matrices = {
    // "thirdparty/1138_bus/1138_bus.mtx";,
    "thirdparty/audikw_1/audikw_1.mtx",
    "thirdparty/F1/F1.mtx",
    "thirdparty/af_shell7/af_shell7.mtx",
    "thirdparty/mario002/mario002.mtx",
    "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
    "thirdparty/msdoor/msdoor.mtx"
  };

  if (rank == 0) {
    std::cout << " ============= BENCHMARK: MPI + OMP GUIDED ============= "
              << std::endl;
    std::cout << " MPI Ranks: " << size << " | OMP Threads: " << omp_threads << std::endl;
  }

  std::string csv_file = "mpi_spmv_results.csv";
  if (rank == 0) {
    std::cout << "--> Initializing CSV: " << csv_file << std::endl;
    SparseMatrixBenchmark::writeMPIcsvHeader(csv_file);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (const auto &path : matrices) {
    // Extract matrix name from path
    std::filesystem::path fs_path(path);
    std::string matrix_name = fs_path.stem().string();

    if (rank == 0) {
      std::cout << "\nProcessing matrix: " << matrix_name << " (" << path << ")" << std::endl;
    }

    COOMatrix global;

    if (rank == 0) {
      try {
        global.readMatrixMarket(path);
        if (rank == 0) {
          std::cout << "  Matrix loaded: " << global.rows << " x " << global.cols
                    << ", nnz = " << global.nnz << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "Error loading matrix " << path << ": " << e.what() << std::endl;
        continue;
      }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&global.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize vectors on non-root ranks
    if (rank != 0) {
      global.row_idx.resize(global.nnz);
      global.col_idx.resize(global.nnz);
      global.values.resize(global.nnz);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast matrix data
    MPI_Bcast(global.row_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global.col_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global.values.data(), global.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> x(global.cols, 1.0);

    if (rank == 0) {
      std::cout << "  Testing 1D partitioning..." << std::endl;
    }

    // 1D Test - DO NOT CLEAR GLOBAL MATRIX HERE!
    DistributedMatrix A1(global, Partitioning::OneD);
    BenchmarkResult res1 = SparseMatrixBenchmark::benchmark_spmv(A1, x, 10);
    if (rank == 0) {
      SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D", size, omp_threads, global.nnz, res1);
      std::cout << "    90th percentile: " << res1.percentile_90 << " ms" << std::endl;
    }

    // Only test 2D if we have more than 1 process
    if (size > 1) {
      if (rank == 0) {
        std::cout << "  Testing 2D partitioning..." << std::endl;
      }

      // 2D Test - NEED THE GLOBAL MATRIX TO STILL BE VALID HERE!
      DistributedMatrix A2(global, Partitioning::TwoD);
      BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x, 10);
      if (rank == 0) {
        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D", size, omp_threads, global.nnz, res2);
        std::cout << "    90th percentile: " << res2.percentile_90 << " ms" << std::endl;
      }
    } else if (rank == 0) {
      std::cout << "  Skipping 2D partitioning (requires >1 MPI process)" << std::endl;
    }

    // Now we can clear the global matrix if we want, but it will go out of scope anyway
    // at the end of this loop iteration, so clearing is unnecessary
  }

  // ============================================================
  //                    WEAK SCALING BENCHMARK
  // ============================================================
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
      std::cout << "\n ============= WEAK SCALING (Random Matrix) ============= " << std::endl;
  }

  // Configuration for Weak Scaling
  int base_dim = 2000;
  int scaled_dim = static_cast<int>(base_dim * std::sqrt(size));

  std::string ws_csv = "mpi_weak_scaling.csv";
  if (rank == 0) SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);

  COOMatrix weak_global;

  if (rank == 0) {
      std::cout << "  Generating Random Matrix for P=" << size
                << " | Dim: " << scaled_dim << "x" << scaled_dim << std::endl;
      weak_global.generateRandomSparse(scaled_dim, 0.99);
  }

  // Broadcast Dimensions
  MPI_Bcast(&weak_global.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&weak_global.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&weak_global.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
      weak_global.row_idx.resize(weak_global.nnz);
      weak_global.col_idx.resize(weak_global.nnz);
      weak_global.values.resize(weak_global.nnz);
  }

  // Broadcast Data
  MPI_Bcast(weak_global.row_idx.data(), weak_global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(weak_global.col_idx.data(), weak_global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(weak_global.values.data(), weak_global.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x_weak(weak_global.cols, 1.0);

  // Run 1D Weak Scaling
  DistributedMatrix A_weak(weak_global, Partitioning::OneD);

  // METRIC: Memory Footprint
  size_t local_mem = A_weak.getLocalMemoryUsage();
  size_t max_mem = 0;
  MPI_Reduce(&local_mem, &max_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  BenchmarkResult res_weak = SparseMatrixBenchmark::benchmark_spmv(A_weak, x_weak, 10);

  if (rank == 0) {
      SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "1D", size, omp_threads, weak_global.nnz, res_weak);
      std::cout << "    Time: " << res_weak.average << " ms | Max Mem per Rank: "
                << (max_mem / 1024.0 / 1024.0) << " MB" << std::endl;
  }

  // Optional: Run 2D Weak Scaling if size > 1
  if (size > 1) {
       DistributedMatrix A_weak_2d(weak_global, Partitioning::TwoD);
       BenchmarkResult res_weak2 = SparseMatrixBenchmark::benchmark_spmv(A_weak_2d, x_weak, 10);
       if (rank == 0) {
           SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "2D", size, omp_threads, weak_global.nnz, res_weak2);
       }
  }

  if (rank == 0) {
    std::cout << "\n ============= BENCHMARK COMPLETE ============= " << std::endl;
  }

  MPI_Finalize();
  return 0;
}
