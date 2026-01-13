#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"

#include <filesystem>
#include <iostream>
#include <cmath>
#include <omp.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    // --parallel-io means: use filename constructor (MPI-IO reader)
    bool use_parallel_io = false;
    bool compare_modes = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--parallel-io") {
            use_parallel_io = true;
        } else if (arg == "--compare") {
            compare_modes = true;
        } else if (arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "Options:\n";
                std::cout << "  --parallel-io : Use MPI-IO reader (filename constructor)\n";
                std::cout << "  --compare     : Compare both modes\n";
                std::cout << "  --help        : Show this help\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        std::cout << " ============= INITIATING BENCHMARK =============\n";
        if (use_parallel_io) {
            std::cout << " Using MPI-IO mode (filename constructor)\n";
        } else {
            std::cout << " Using rank0-read + distribute mode\n";
        }
        if (compare_modes) {
            std::cout << " Comparing both modes\n";
        }
    }

    // Get OpenMP thread count
    int omp_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        omp_threads = omp_get_num_threads();
    }

    std::vector<std::string> matrices = {
        // "thirdparty/1138_bus/1138_bus.mtx",
        "thirdparty/audikw_1/audikw_1.mtx",
        "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
        "thirdparty/Serena/Serena.mtx",
        "thirdparty/Freescale1/Freescale1.mtx",
        "thirdparty/ldoor/ldoor.mtx",
        "thirdparty/G3_circuit/G3_circuit.mtx",
        "thirdparty/Transport/Transport.mtx"
    };

    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK CONFIGURATION =============\n";
        std::cout << " MPI Ranks: " << size << " | OMP Threads: " << omp_threads << "\n";
        std::cout << " Matrices to test: " << matrices.size() << "\n";
    }

    const std::string csv_file = "mpi_spmv_results.csv";
    if (rank == 0) {
        std::cout << "--> Results file: " << csv_file << "\n";
        SparseMatrixBenchmark::writeMPIcsvHeader(csv_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (const auto &path : matrices) {
        std::filesystem::path fs_path(path);
        std::string matrix_name = fs_path.stem().string();

        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "Matrix: " << matrix_name << "\n";
            std::cout << "========================================\n";
        }

        // ============================================================
        // MODE A: rank0 reads full matrix, constructor distributes nnz
        // ============================================================
        if (compare_modes || !use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- rank0-read + distribute ---\n";
            }

            COOMatrix global; // only valid on rank0
            bool ok = true;
            if (rank == 0) {
                try {
                    global.readMatrixMarket(path);
                    std::cout << "  Loaded by rank 0: " << global.rows << " x " << global.cols
                              << ", nnz = " << global.nnz << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "Error loading matrix: " << e.what() << "\n";
                    ok = false;
                }
            }
            MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if (!ok) {
                MPI_Barrier(MPI_COMM_WORLD);
            } else {
                // IMPORTANT: do not broadcast COO arrays. Constructor will broadcast dims and distribute nnz.
                DistributedMatrix A1(global, Partitioning::OneD, MPI_COMM_WORLD, false);
                std::vector<double> x(A1.global_cols, 1.0);

                size_t local_mem = A1.getLocalMemoryUsage();
                size_t max_mem = 0;
                MPI_Reduce(&local_mem, &max_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb = max_mem / (1024.0 * 1024.0);

                BenchmarkResult res = SparseMatrixBenchmark::benchmark_spmv(A1, x, 10);

                if (rank == 0) {
                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                          size, omp_threads, global.nnz,
                                                          mem_mb, res);
                }

                if (size > 1) {
                    DistributedMatrix A2(global, Partitioning::TwoD, MPI_COMM_WORLD, false);
                    std::vector<double> x2(A2.global_cols, 1.0);

                    size_t local_mem2 = A2.getLocalMemoryUsage();
                    size_t max_mem2 = 0;
                    MPI_Reduce(&local_mem2, &max_mem2, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                    double mem_mb2 = max_mem2 / (1024.0 * 1024.0);

                    BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x2, 10);
                    if (rank == 0) {
                        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                              size, omp_threads, global.nnz,
                                                              mem_mb2, res2);
                    }
                }
            }
        }

        // ============================================================
        // MODE B: MPI-IO reader (filename constructor)
        // ============================================================
        if (compare_modes || use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- MPI-IO (filename constructor) ---\n";
            }

            bool ok = true;
            DistributedMatrix *A1 = nullptr;
            DistributedMatrix *A2 = nullptr;

            try {
                A1 = new DistributedMatrix(path, Partitioning::OneD, MPI_COMM_WORLD);
                std::vector<double> x(A1->global_cols, 1.0);

                size_t local_mem = A1->getLocalMemoryUsage();
                size_t max_mem = 0;
                MPI_Reduce(&local_mem, &max_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb = max_mem / (1024.0 * 1024.0);

                BenchmarkResult res = SparseMatrixBenchmark::benchmark_spmv(*A1, x, 10);

                unsigned long long local_nnz = (unsigned long long)A1->local_csr.nnz;
                unsigned long long total_nnz = 0;
                MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

                if (rank == 0) {
                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                          size, omp_threads, (int)total_nnz,
                                                          mem_mb, res);
                }

                if (size > 1) {
                    A2 = new DistributedMatrix(path, Partitioning::TwoD, MPI_COMM_WORLD);
                    std::vector<double> x2(A2->global_cols, 1.0);

                    size_t local_mem2 = A2->getLocalMemoryUsage();
                    size_t max_mem2 = 0;
                    MPI_Reduce(&local_mem2, &max_mem2, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                    double mem_mb2 = max_mem2 / (1024.0 * 1024.0);

                    BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(*A2, x2, 10);

                    unsigned long long local_nnz2 = (unsigned long long)A2->local_csr.nnz;
                    unsigned long long total_nnz2 = 0;
                    MPI_Allreduce(&local_nnz2, &total_nnz2, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

                    if (rank == 0) {
                        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                              size, omp_threads, (int)total_nnz2,
                                                              mem_mb2, res2);
                    }
                }
            } catch (const std::exception& e) {
                ok = false;
                if (rank == 0) {
                    std::cerr << "MPI-IO constructor error: " << e.what() << "\n";
                }
            }

            delete A1;
            delete A2;

            MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ============================================================
    // WEAK SCALING (Random Matrix)
    // ============================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n ============= WEAK SCALING (Random Matrix) =============\n";
    }

    int base_nnz_per_proc = 1000000;
    int total_nnz_weak = base_nnz_per_proc * size;
    double density = 0.001;
    int weak_dim = (int)std::sqrt(total_nnz_weak / density);

    std::string ws_csv = "mpi_weak_scaling.csv";
    if (rank == 0) {
        SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);
    }

    bool weak_ok = true;
    COOMatrix weak_global; // only rank0

    if (rank == 0) {
        try {
            weak_global.generateRandomSparseNNZ(weak_dim, density, total_nnz_weak);
        } catch (const std::exception& e) {
            std::cerr << "Weak scaling matrix generation error: " << e.what() << "\n";
            weak_ok = false;
        }
    }
    MPI_Bcast(&weak_ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (weak_ok) {
        DistributedMatrix A1(weak_global, Partitioning::OneD, MPI_COMM_WORLD, false);
        std::vector<double> x(A1.global_cols, 1.0);

        size_t local_mem = A1.getLocalMemoryUsage();
        size_t max_mem = 0;
        MPI_Reduce(&local_mem, &max_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        double mem_mb = max_mem / (1024.0 * 1024.0);

        BenchmarkResult res = SparseMatrixBenchmark::benchmark_spmv(A1, x, 10);

        int weak_nnz = 0;
        if (rank == 0) weak_nnz = weak_global.nnz;
        MPI_Bcast(&weak_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "1D",
                                                  size, omp_threads, weak_nnz,
                                                  mem_mb, res);
        }

        if (size > 1) {
            DistributedMatrix A2(weak_global, Partitioning::TwoD, MPI_COMM_WORLD, false);
            std::vector<double> x2(A2.global_cols, 1.0);

            size_t local_mem2 = A2.getLocalMemoryUsage();
            size_t max_mem2 = 0;
            MPI_Reduce(&local_mem2, &max_mem2, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double mem_mb2 = max_mem2 / (1024.0 * 1024.0);

            BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x2, 10);

            if (rank == 0) {
                SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "2D",
                                                      size, omp_threads, weak_nnz,
                                                      mem_mb2, res2);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK COMPLETE =============\n";
        std::cout << "Strong scaling: " << csv_file << "\n";
        if (weak_ok) std::cout << "Weak scaling:   " << ws_csv << "\n";
    }

    MPI_Finalize();
    return 0;
}
