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
    // "thirdparty/1138_bus/1138_bus.mtx",
    "thirdparty/F1/F1.mtx",
    "thirdparty/af_shell7/af_shell7.mtx",
    "thirdparty/mario002/mario002.mtx",
    "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
    "thirdparty/msdoor/msdoor.mtx"
  };
  // std::vector<std::string> matrices = {
    // "thirdparty/bcsstk36/bcsstk36.mtx",
    // "thirdparty/bcsstk30/bcsstk30.mtx",
    // "thirdparty/rdb968/rdb968.mtx",
    // "thirdparty/bcsstk25/bcsstk25.mtx",
    // "thirdparty/af23560/af23560.mtx"
  // };
  //
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

    COOMatrix global;

    if (rank == 0) {
      global.readMatrixMarket(path);
    }

    MPI_Bcast(&global.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      global.row_idx.resize(global.nnz);
      global.col_idx.resize(global.nnz);
      global.values.resize(global.nnz);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(global.row_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global.col_idx.data(), global.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global.values.data(), global.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> x(global.cols, 1.0);
    // 1D Test
    DistributedMatrix A1(global, Partitioning::OneD);
    BenchmarkResult res1 = SparseMatrixBenchmark::benchmark_spmv(A1, x, 10);
    if (rank == 0) {
      SparseMatrixBenchmark::writeMPIcsvRow(csv_file, "1138_bus", "1D", size, omp_threads, global.nnz, res1);
    }

    // 2D Test
    DistributedMatrix A2(global, Partitioning::TwoD);
    BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x, 10);
    if (rank == 0) {
      SparseMatrixBenchmark::writeMPIcsvRow(csv_file, "1138_bus", "2D", size, omp_threads, global.nnz, res2);
    }
  }

  MPI_Finalize();
  return 0;
}
