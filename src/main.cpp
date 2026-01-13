// main.cpp - FIXED VERSION
#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    bool use_parallel_io = false;   // uses filename constructor (rank0 read + scatter in reader)
    bool compare_modes  = false;

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
                std::cout << "  --parallel-io : Use filename constructor (rank0 read + scatter)\n";
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
            std::cout << " Using PARALLEL-I/O constructor mode (rank0 read + scatter)\n";
        } else {
            std::cout << " Using SEQUENTIAL I/O mode (rank0 read + scatter)\n";
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
        "thirdparty/1138_bus/1138_bus.mtx",
        // "thirdparty/audikw_1/audikw_1.mtx",
        // "thirdparty/kron_g500-logn19/kron_g500-logn19.mtx",
        // "thirdparty/Serena/Serena.mtx",
        // "thirdparty/Freescale1/Freescale1.mtx",
        // "thirdparty/ldoor/ldoor.mtx",
        // "thirdparty/G3_circuit/G3_circuit.mtx",
        // "thirdparty/Transport/Transport.mtx"
    };

    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK CONFIGURATION =============\n";
        std::cout << " MPI Ranks: " << size << " | OMP Threads: " << omp_threads << "\n";
        std::cout << " Matrices to test: " << matrices.size() << "\n";
    }

    // Single CSV file
    std::string csv_file = "mpi_spmv_results.csv";

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
        // MODE A: "SEQUENTIAL I/O" (rank0 reads full matrix, ctor scatters nnz)
        // ============================================================
        if (compare_modes || !use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- SEQUENTIAL I/O (rank0 read + scatter) ---\n";
            }

            COOMatrix global_seq; // valid only on rank 0

            bool ok = true;
            if (rank == 0) {
                try {
                    global_seq.readMatrixMarket(path);
                    std::cout << "  Loaded by rank 0: " << global_seq.rows << " x "
                              << global_seq.cols << ", nnz = " << global_seq.nnz << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "Error loading matrix: " << e.what() << "\n";
                    ok = false;
                }
            }
            MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if (!ok) {
                MPI_Barrier(MPI_COMM_WORLD);
                continue;
            }

            // IMPORTANT: do NOT broadcast COO arrays anymore.
            // The constructor will broadcast dims and scatter nonzeros.
            DistributedMatrix A1_seq(global_seq, Partitioning::OneD, MPI_COMM_WORLD, false);

            std::vector<double> x1(A1_seq.global_cols, 1.0);

            // Memory usage (max across ranks)
            size_t local_mem_bytes_1 = A1_seq.getLocalMemoryUsage();
            size_t max_mem_bytes_1 = 0;
            MPI_Reduce(&local_mem_bytes_1, &max_mem_bytes_1, 1,
                       MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double mem_mb_1 = max_mem_bytes_1 / (1024.0 * 1024.0);

            BenchmarkResult res1_seq = SparseMatrixBenchmark::benchmark_spmv(A1_seq, x1, 10);

            if (rank == 0) {
                std::cout << "  1D Results:\n";
                std::cout << "    Avg time: " << res1_seq.average << " ms\n";
                std::cout << "    Max memory: " << mem_mb_1 << " MB\n";

                SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                      size, omp_threads, global_seq.nnz,
                                                      mem_mb_1, res1_seq);
            }

            if (size > 1) {
                DistributedMatrix A2_seq(global_seq, Partitioning::TwoD, MPI_COMM_WORLD, false);

                std::vector<double> x2(A2_seq.global_cols, 1.0);

                size_t local_mem_bytes_2 = A2_seq.getLocalMemoryUsage();
                size_t max_mem_bytes_2 = 0;
                MPI_Reduce(&local_mem_bytes_2, &max_mem_bytes_2, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb_2 = max_mem_bytes_2 / (1024.0 * 1024.0);

                BenchmarkResult res2_seq = SparseMatrixBenchmark::benchmark_spmv(A2_seq, x2, 10);

                if (rank == 0) {
                    std::cout << "  2D Results:\n";
                    std::cout << "    Avg time: " << res2_seq.average << " ms\n";
                    std::cout << "    Max memory: " << mem_mb_2 << " MB\n";

                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                          size, omp_threads, global_seq.nnz,
                                                          mem_mb_2, res2_seq);
                }
            }
        }

        // ============================================================
        // MODE B: "PARALLEL I/O constructor" (filename ctor -> reader -> scatter)
        // NOTE: This is NOT MPI-IO; it uses rank0 read + scatter in SimpleParallelReader.
        // ============================================================
        if (compare_modes || use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- FILENAME CONSTRUCTOR (rank0 read + scatter) ---\n";
            }

            bool ok = true;

            DistributedMatrix *A1_par = nullptr;
            DistributedMatrix *A2_par = nullptr;

            try {
                A1_par = new DistributedMatrix(path, Partitioning::OneD, MPI_COMM_WORLD);

                std::vector<double> x(A1_par->global_cols, 1.0);

                size_t local_mem_bytes_1 = A1_par->getLocalMemoryUsage();
                size_t max_mem_bytes_1 = 0;
                MPI_Reduce(&local_mem_bytes_1, &max_mem_bytes_1, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb_1 = max_mem_bytes_1 / (1024.0 * 1024.0);

                BenchmarkResult res1_par = SparseMatrixBenchmark::benchmark_spmv(*A1_par, x, 10);

                // We want nnz in CSV: in this mode rank0 doesn’t hold a "global COO" variable.
                // So we compute total nnz as sum of local nnz:
                unsigned long long local_nnz = (unsigned long long)A1_par->local_csr.nnz;
                unsigned long long total_nnz = 0;
                MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

                if (rank == 0) {
                    std::cout << "  1D Results:\n";
                    std::cout << "    Avg time: " << res1_par.average << " ms\n";
                    std::cout << "    Max memory: " << mem_mb_1 << " MB\n";
                    std::cout << "    Total nnz (sum local): " << total_nnz << "\n";

                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                          size, omp_threads, (int)total_nnz,
                                                          mem_mb_1, res1_par);
                }

                if (size > 1) {
                    A2_par = new DistributedMatrix(path, Partitioning::TwoD, MPI_COMM_WORLD);

                    std::vector<double> x2(A2_par->global_cols, 1.0);

                    size_t local_mem_bytes_2 = A2_par->getLocalMemoryUsage();
                    size_t max_mem_bytes_2 = 0;
                    MPI_Reduce(&local_mem_bytes_2, &max_mem_bytes_2, 1,
                               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                    double mem_mb_2 = max_mem_bytes_2 / (1024.0 * 1024.0);

                    BenchmarkResult res2_par = SparseMatrixBenchmark::benchmark_spmv(*A2_par, x2, 10);

                    unsigned long long local_nnz2 = (unsigned long long)A2_par->local_csr.nnz;
                    unsigned long long total_nnz2 = 0;
                    MPI_Allreduce(&local_nnz2, &total_nnz2, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

                    if (rank == 0) {
                        std::cout << "  2D Results:\n";
                        std::cout << "    Avg time: " << res2_par.average << " ms\n";
                        std::cout << "    Max memory: " << mem_mb_2 << " MB\n";
                        std::cout << "    Total nnz (sum local): " << total_nnz2 << "\n";

                        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                              size, omp_threads, (int)total_nnz2,
                                                              mem_mb_2, res2_par);
                    }
                }

            } catch (const std::exception &e) {
                ok = false;
                if (rank == 0) {
                    std::cerr << "  Constructor-mode error: " << e.what() << "\n";
                }
            }

            // cleanup
            delete A1_par;
            delete A2_par;

            // ensure all ranks stay aligned
            MPI_Bcast(&ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ============================================================
    // WEAK SCALING BENCHMARK (Random Matrix)
    // FIX: do NOT broadcast full COO; rank0 generates and ctor scatters.
    // ============================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n ============= WEAK SCALING (Random Matrix) =============\n";
    }

    int base_nnz_per_proc = 1000000;
    int total_nnz_weak = base_nnz_per_proc * size;
    double density = 0.001;
    int weak_dim = static_cast<int>(std::sqrt(total_nnz_weak / density));

    std::string ws_csv = "mpi_weak_scaling.csv";
    if (rank == 0) {
        std::cout << "  Weak Scaling Configuration:\n";
        std::cout << "    Processors: " << size << "\n";
        std::cout << "    Base nnz per processor: " << base_nnz_per_proc << "\n";
        std::cout << "    Target total nnz: " << total_nnz_weak << "\n";
        std::cout << "    Matrix dimension (approx): " << weak_dim << "x" << weak_dim << "\n";
        SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);
    }

    bool weak_scaling_success = true;
    COOMatrix weak_global; // valid only on rank0

    if (rank == 0) {
        std::cout << "  Generating Random Sparse Matrix...\n";
        try {
            weak_global.generateRandomSparseNNZ(weak_dim, density, total_nnz_weak);
            std::cout << "    Actual nnz generated: " << weak_global.nnz << "\n";
            std::cout << "    Actual density: "
                      << (100.0 * weak_global.nnz) / (weak_global.rows * weak_global.cols)
                      << "%\n";
        } catch (const std::exception& e) {
            std::cerr << "Error generating weak scaling matrix: " << e.what() << "\n";
            weak_scaling_success = false;
        }
    }

    MPI_Bcast(&weak_scaling_success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (weak_scaling_success) {
        // 1D weak scaling
        if (rank == 0) {
            std::cout << "\n--- Weak Scaling: 1D Partitioning ---\n";
        }

        DistributedMatrix A_weak_1d(weak_global, Partitioning::OneD, MPI_COMM_WORLD, false);
        std::vector<double> x_weak(A_weak_1d.global_cols, 1.0);

        size_t local_mem_1d = A_weak_1d.getLocalMemoryUsage();
        size_t max_mem_1d = 0;
        MPI_Reduce(&local_mem_1d, &max_mem_1d, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        double max_mem_mb_1d = max_mem_1d / (1024.0 * 1024.0);

        BenchmarkResult res_weak_1d = SparseMatrixBenchmark::benchmark_spmv(A_weak_1d, x_weak, 10);

        // total nnz from rank0 (only rank0 has weak_global.nnz valid)
        int weak_nnz = 0;
        if (rank == 0) weak_nnz = weak_global.nnz;
        MPI_Bcast(&weak_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "1D",
                                                 size, omp_threads, weak_nnz,
                                                 max_mem_mb_1d, res_weak_1d);
            std::cout << "    Avg time: " << res_weak_1d.average << " ms\n";
            std::cout << "    Max memory per rank: " << max_mem_mb_1d << " MB\n";
        }

        // 2D weak scaling
        if (size > 1) {
            if (rank == 0) {
                std::cout << "\n--- Weak Scaling: 2D Partitioning ---\n";
            }

            DistributedMatrix A_weak_2d(weak_global, Partitioning::TwoD, MPI_COMM_WORLD, false);
            std::vector<double> x_weak2(A_weak_2d.global_cols, 1.0);

            size_t local_mem_2d = A_weak_2d.getLocalMemoryUsage();
            size_t max_mem_2d = 0;
            MPI_Reduce(&local_mem_2d, &max_mem_2d, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double max_mem_mb_2d = max_mem_2d / (1024.0 * 1024.0);

            BenchmarkResult res_weak_2d = SparseMatrixBenchmark::benchmark_spmv(A_weak_2d, x_weak2, 10);

            if (rank == 0) {
                SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "Random_Weak", "2D",
                                                     size, omp_threads, weak_nnz,
                                                     max_mem_mb_2d, res_weak_2d);
                std::cout << "    Avg time: " << res_weak_2d.average << " ms\n";
                std::cout << "    Max memory per rank: " << max_mem_mb_2d << " MB\n";
            }
        }

        if (rank == 0) {
            std::cout << "\nWeak scaling results saved to: " << ws_csv << "\n";
        }
    } else if (rank == 0) {
        std::cout << "Skipping weak scaling due to matrix generation error.\n";
    }

    // ============================================================
    // FINALIZATION
    // ============================================================
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK COMPLETE =============\n";
        std::cout << "Results saved to:\n";
        std::cout << "  - Strong scaling: " << csv_file << "\n";
        if (weak_scaling_success) {
            std::cout << "  - Weak scaling: " << ws_csv << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}

