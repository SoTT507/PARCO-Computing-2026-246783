#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"

int main(int argc, char **argv) {
  // region: Deliverable 1 main code
  //  SparseMatrixBenchmark benchmark;

  // benchmark.addMatrixFile("thirdparty/F1/F1.mtx");                 //~340k
  // rows benchmark.addMatrixFile("thirdparty/af_shell7/af_shell7.mtx");   //~5M
  // rows benchmark.addMatrixFile("thirdparty/mario002/mario002.mtx"); //~2M
  // rows
  // benchmark.addMatrixFile("thirdparty/kron_g500-logn19/kron_g500-logn19.mtx");
  // //~524k rows benchmark.addMatrixFile("thirdparty/msdoor/msdoor.mtx");
  // //~944k rows

  // benchmark.addMatrixFile("thirdparty/bcsstk36/bcsstk36.mtx"); //~23k rows
  // benchmark.addMatrixFile("thirdparty/bcsstk30/bcsstk30.mtx");//~34k rows
  // benchmark.addMatrixFile("thirdparty/rdb968/rdb968.mtx");//~5k --> LOW
  // sparsity benchmark.addMatrixFile("thirdparty/bcsstk25/bcsstk25.mtx");
  // benchmark.addMatrixFile("thirdparty/af23560/af23560.mtx");
  //
  //
  // Loop to run warmup directly from program
  // instead of running the program multiple times
  //
  // for(int i = 1; i<=5; i++){
  // benchmark.warmup(); //same as runFullBenchmark but does not save nor print
  // results std::cout<< "Warmup run: " << i << std::endl;
  // }

  // Run full benchmark
  // benchmark.runFullBenchmark();
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
    "thirdparty/1138_bus/1138_bus.mtx",
    // "thirdparty/F1/F1.mtx",
    // "thirdparty/af_shell7/af_shell7.mtx",
    // "thirdparty/mario002/mario002.mtx",
    // "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
    // "thirdparty/msdoor/msdoor.mtx"
  };
  // std::vector<std::string> matrices = {
    // "thirdparty/bcsstk36/bcsstk36.mtx",
    // "thirdparty/bcsstk30/bcsstk30.mtx",
    // "thirdparty/rdb968/rdb968.mtx",
    // "thirdparty/bcsstk25/bcsstk25.mtx",
    // "thirdparty/af23560/af23560.mtx"
  // };
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
    std::string matrix_name = fs_path.stem().string();  // e.g., "1138_bus", "F1", etc.
    
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
        // Continue to next matrix
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
    
    // 1D Test
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
      
      // 2D Test
      DistributedMatrix A2(global, Partitioning::TwoD);
      BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x, 10);
      if (rank == 0) {
        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D", size, omp_threads, global.nnz, res2);
        std::cout << "    90th percentile: " << res2.percentile_90 << " ms" << std::endl;
      }
    } else if (rank == 0) {
      std::cout << "  Skipping 2D partitioning (requires >1 MPI process)" << std::endl;
    }
  }

  if (rank == 0) {
    std::cout << "\n ============= BENCHMARK COMPLETE ============= " << std::endl;
    std::cout << "Results saved to: " << csv_file << std::endl;
  }

  MPI_Finalize();
  return 0;
}
