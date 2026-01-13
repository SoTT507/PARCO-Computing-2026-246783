// main.cpp - FIXED VERSION
#include "d_matrix.hpp"
#include "benchmark.hpp"
#include "pch.h"
#include <filesystem>
#include <iostream>
#include <iomanip>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
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
                std::cout << "  --parallel-io : Use parallel file reading\n";
                std::cout << "  --compare     : Compare sequential vs parallel I/O\n";
                std::cout << "  --help        : Show this help\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        std::cout << " ============= INITIATING BENCHMARK ============= " << std::endl;
        if (use_parallel_io) {
            std::cout << " Using PARALLEL I/O mode (actually using new constructor)" << std::endl;
        } else {
            std::cout << " Using SEQUENTIAL I/O mode" << std::endl;
        }
        if (compare_modes) {
            std::cout << " Comparing both I/O modes" << std::endl;
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
        std::cout << "\n ============= BENCHMARK CONFIGURATION =============" << std::endl;
        std::cout << " MPI Ranks: " << size << " | OMP Threads: " << omp_threads << std::endl;
        std::cout << " Matrices to test: " << matrices.size() << std::endl;
    }

    // Single CSV file (as per your request)
    std::string csv_file = "mpi_spmv_results.csv";

    // Write header once
    if (rank == 0) {
        std::cout << "--> Results file: " << csv_file << std::endl;
        SparseMatrixBenchmark::writeMPIcsvHeader(csv_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (const auto &path : matrices) {
        std::filesystem::path fs_path(path);
        std::string matrix_name = fs_path.stem().string();

        if (rank == 0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Matrix: " << matrix_name << std::endl;
            std::cout << "========================================" << std::endl;
        }

        // ============================================================
        // SEQUENTIAL I/O (original method - load on rank 0 then broadcast)
        // ============================================================
        if (compare_modes || !use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- SEQUENTIAL I/O ---" << std::endl;
            }

            COOMatrix global_seq;
            
            if (rank == 0) {
                try {
                    global_seq.readMatrixMarket(path);
                    std::cout << "  Sequentially loaded by rank 0: " << global_seq.rows << " x "
                              << global_seq.cols << ", nnz = " << global_seq.nnz << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error loading matrix: " << e.what() << std::endl;
                    continue;
                }
            }

            // Broadcast dimensions
            MPI_Bcast(&global_seq.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&global_seq.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&global_seq.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Allocate on non-root ranks
            if (rank != 0) {
                global_seq.row_idx.resize(global_seq.nnz);
                global_seq.col_idx.resize(global_seq.nnz);
                global_seq.values.resize(global_seq.nnz);
            }

            // Broadcast matrix data
            MPI_Bcast(global_seq.row_idx.data(), global_seq.nnz, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(global_seq.col_idx.data(), global_seq.nnz, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(global_seq.values.data(), global_seq.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> x_global(global_seq.cols, 1.0);

            // Test 1D Partitioning (Sequential)
            if (rank == 0) {
                std::cout << "  Testing 1D Partitioning (Sequential)..." << std::endl;
            }

            DistributedMatrix A1_seq(global_seq, Partitioning::OneD, MPI_COMM_WORLD);

            size_t local_mem_bytes_1 = A1_seq.getLocalMemoryUsage();
            size_t max_mem_bytes_1 = 0;
            MPI_Reduce(&local_mem_bytes_1, &max_mem_bytes_1, 1,
                       MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double mem_mb_1 = max_mem_bytes_1 / (1024.0 * 1024.0);

            BenchmarkResult res1_seq = SparseMatrixBenchmark::benchmark_spmv(A1_seq, x_global, 10);

            if (rank == 0) {
                std::cout << "    1D Sequential Results:" << std::endl;
                std::cout << "      Avg time: " << res1_seq.average << " ms" << std::endl;
                std::cout << "      Max memory: " << mem_mb_1 << " MB" << std::endl;

                SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                      size, omp_threads, global_seq.nnz,
                                                      mem_mb_1, res1_seq);
            }

            // Test 2D Partitioning if size > 1
            if (size > 1) {
                if (rank == 0) {
                    std::cout << "  Testing 2D Partitioning (Sequential)..." << std::endl;
                }

                DistributedMatrix A2_seq(global_seq, Partitioning::TwoD, MPI_COMM_WORLD);

                size_t local_mem_bytes_2 = A2_seq.getLocalMemoryUsage();
                size_t max_mem_bytes_2 = 0;
                MPI_Reduce(&local_mem_bytes_2, &max_mem_bytes_2, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb_2 = max_mem_bytes_2 / (1024.0 * 1024.0);

                BenchmarkResult res2_seq = SparseMatrixBenchmark::benchmark_spmv(A2_seq, x_global, 10);

                if (rank == 0) {
                    std::cout << "    2D Sequential Results:" << std::endl;
                    std::cout << "      Avg time: " << res2_seq.average << " ms" << std::endl;
                    std::cout << "      Max memory: " << mem_mb_2 << " MB" << std::endl;

                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                          size, omp_threads, global_seq.nnz,
                                                          mem_mb_2, res2_seq);
                }
            }
        }

        // ============================================================
        // PARALLEL I/O (new method - each process reads its portion)
        // ============================================================
        if (compare_modes || use_parallel_io) {
            if (rank == 0 && compare_modes) {
                std::cout << "\n--- PARALLEL I/O ---" << std::endl;
            }

            if (rank == 0) {
                std::cout << "  Testing parallel file reading..." << std::endl;
            }

            // Try parallel I/O, fallback to sequential if it fails
            bool parallel_success = true;
            COOMatrix global_par;
            
            // First, try to get metadata to know the real nnz
            int actual_nnz = 0;
            try {
                if (rank == 0) {
                    // Read just the metadata to get actual nnz
                    COOMatrix meta_check;
                    meta_check.readMatrixMarket(path);
                    actual_nnz = meta_check.nnz;
                    std::cout << "  Matrix has " << actual_nnz << " non-zeros" << std::endl;
                }
                MPI_Bcast(&actual_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
            } catch (...) {
                parallel_success = false;
            }

            // Now try parallel reading
            DistributedMatrix* A1_par = nullptr;
            DistributedMatrix* A2_par = nullptr;
            
            try {
                if (rank == 0) {
                    std::cout << "  Attempting parallel 1D partitioning..." << std::endl;
                }
                
                // Use parallel constructor
                A1_par = new DistributedMatrix(path, Partitioning::OneD, MPI_COMM_WORLD);
                
                // Validate: check if local CSR has reasonable nnz
                size_t local_nnz = A1_par->local_csr.nnz;
                size_t total_nnz = 0;
                MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                
                if (rank == 0) {
                    std::cout << "  Parallel reading reports " << total_nnz << " total non-zeros" << std::endl;
                    std::cout << "  Actual matrix has " << actual_nnz << " non-zeros" << std::endl;
                }
                
                // If the reported nnz is wildly different from actual, parallel reading failed
                if (total_nnz == 0 || abs((long long)total_nnz - (long long)actual_nnz) > actual_nnz * 0.5) {
                    if (rank == 0) {
                        std::cout << "  Parallel I/O validation failed (nnz mismatch)" << std::endl;
                        std::cout << "  Falling back to sequential I/O..." << std::endl;
                    }
                    parallel_success = false;
                    delete A1_par;
                    A1_par = nullptr;
                }
            } catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "  Parallel I/O error: " << e.what() << std::endl;
                    std::cout << "  Falling back to sequential I/O..." << std::endl;
                }
                parallel_success = false;
                if (A1_par) {
                    delete A1_par;
                    A1_par = nullptr;
                }
            }

            if (parallel_success && A1_par) {
                // Parallel I/O succeeded - run benchmarks
                std::vector<double> x_global_par(A1_par->global_cols, 1.0);

                size_t local_mem_bytes_1_par = A1_par->getLocalMemoryUsage();
                size_t max_mem_bytes_1_par = 0;
                MPI_Reduce(&local_mem_bytes_1_par, &max_mem_bytes_1_par, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb_1_par = max_mem_bytes_1_par / (1024.0 * 1024.0);

                BenchmarkResult res1_par = SparseMatrixBenchmark::benchmark_spmv(*A1_par, x_global_par, 10);

                if (rank == 0) {
                    std::cout << "    1D Parallel I/O Results:" << std::endl;
                    std::cout << "      Avg time: " << res1_par.average << " ms" << std::endl;
                    std::cout << "      Max memory: " << mem_mb_1_par << " MB" << std::endl;

                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                          size, omp_threads, actual_nnz,
                                                          mem_mb_1_par, res1_par);
                }

                // Test 2D Partitioning with Parallel I/O if size > 1
                if (size > 1) {
                    try {
                        if (rank == 0) {
                            std::cout << "  Attempting parallel 2D partitioning..." << std::endl;
                        }
                        
                        A2_par = new DistributedMatrix(path, Partitioning::TwoD, MPI_COMM_WORLD);
                        
                        size_t local_mem_bytes_2_par = A2_par->getLocalMemoryUsage();
                        size_t max_mem_bytes_2_par = 0;
                        MPI_Reduce(&local_mem_bytes_2_par, &max_mem_bytes_2_par, 1,
                                   MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                        double mem_mb_2_par = max_mem_bytes_2_par / (1024.0 * 1024.0);

                        BenchmarkResult res2_par = SparseMatrixBenchmark::benchmark_spmv(*A2_par, x_global_par, 10);

                        if (rank == 0) {
                            std::cout << "    2D Parallel I/O Results:" << std::endl;
                            std::cout << "      Avg time: " << res2_par.average << " ms" << std::endl;
                            std::cout << "      Max memory: " << mem_mb_2_par << " MB" << std::endl;

                            SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                                  size, omp_threads, actual_nnz,
                                                                  mem_mb_2_par, res2_par);
                        }
                    } catch (const std::exception& e) {
                        if (rank == 0) {
                            std::cout << "  Parallel 2D failed, skipping..." << std::endl;
                        }
                    }
                }
                
                // Cleanup
                delete A1_par;
                if (A2_par) delete A2_par;
            } else {
                // Parallel I/O failed - fallback to sequential for this mode
                if (rank == 0) {
                    std::cout << "  Using sequential I/O as fallback for parallel mode..." << std::endl;
                }
                
                // Load matrix sequentially
                COOMatrix global_fallback;
                
                if (rank == 0) {
                    try {
                        global_fallback.readMatrixMarket(path);
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading matrix: " << e.what() << std::endl;
                        continue;
                    }
                }

                // Broadcast dimensions
                MPI_Bcast(&global_fallback.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&global_fallback.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&global_fallback.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Allocate on non-root ranks
                if (rank != 0) {
                    global_fallback.row_idx.resize(global_fallback.nnz);
                    global_fallback.col_idx.resize(global_fallback.nnz);
                    global_fallback.values.resize(global_fallback.nnz);
                }

                // Broadcast matrix data
                MPI_Bcast(global_fallback.row_idx.data(), global_fallback.nnz, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(global_fallback.col_idx.data(), global_fallback.nnz, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(global_fallback.values.data(), global_fallback.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                std::vector<double> x_global_fallback(global_fallback.cols, 1.0);

                // Run benchmarks with sequential fallback
                DistributedMatrix A1_fallback(global_fallback, Partitioning::OneD, MPI_COMM_WORLD);

                size_t local_mem_bytes_1_fb = A1_fallback.getLocalMemoryUsage();
                size_t max_mem_bytes_1_fb = 0;
                MPI_Reduce(&local_mem_bytes_1_fb, &max_mem_bytes_1_fb, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                double mem_mb_1_fb = max_mem_bytes_1_fb / (1024.0 * 1024.0);

                BenchmarkResult res1_fb = SparseMatrixBenchmark::benchmark_spmv(A1_fallback, x_global_fallback, 10);

                if (rank == 0) {
                    std::cout << "    1D Fallback Results:" << std::endl;
                    std::cout << "      Avg time: " << res1_fb.average << " ms" << std::endl;
                    std::cout << "      Max memory: " << mem_mb_1_fb << " MB" << std::endl;

                    SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "1D",
                                                          size, omp_threads, global_fallback.nnz,
                                                          mem_mb_1_fb, res1_fb);
                }

                if (size > 1) {
                    DistributedMatrix A2_fallback(global_fallback, Partitioning::TwoD, MPI_COMM_WORLD);

                    size_t local_mem_bytes_2_fb = A2_fallback.getLocalMemoryUsage();
                    size_t max_mem_bytes_2_fb = 0;
                    MPI_Reduce(&local_mem_bytes_2_fb, &max_mem_bytes_2_fb, 1,
                               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
                    double mem_mb_2_fb = max_mem_bytes_2_fb / (1024.0 * 1024.0);

                    BenchmarkResult res2_fb = SparseMatrixBenchmark::benchmark_spmv(A2_fallback, x_global_fallback, 10);

                    if (rank == 0) {
                        std::cout << "    2D Fallback Results:" << std::endl;
                        std::cout << "      Avg time: " << res2_fb.average << " ms" << std::endl;
                        std::cout << "      Max memory: " << mem_mb_2_fb << " MB" << std::endl;

                        SparseMatrixBenchmark::writeMPIcsvRow(csv_file, matrix_name, "2D",
                                                              size, omp_threads, global_fallback.nnz,
                                                              mem_mb_2_fb, res2_fb);
                    }
                }
            }
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

    // Configuration for Weak Scaling
    int base_nnz_per_proc = 1000000;
    int total_nnz_weak = base_nnz_per_proc * size;
    double density = 0.001;
    int weak_dim = static_cast<int>(sqrt(total_nnz_weak / density));

    std::string ws_csv = "mpi_weak_scaling.csv";
    if (rank == 0) {
        std::cout << "  Weak Scaling Configuration:" << std::endl;
        std::cout << "    Processors: " << size << std::endl;
        std::cout << "    Base nnz per processor: " << base_nnz_per_proc << std::endl;
        std::cout << "    Total nnz: " << total_nnz_weak << std::endl;
        std::cout << "    Matrix dimension: " << weak_dim << "x" << weak_dim << std::endl;
        SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);
    }

    bool weak_scaling_success = true;
    COOMatrix weak_global;

    if (rank == 0) {
        std::cout << "  Generating Random Sparse Matrix..." << std::endl;
        try {
            weak_global.generateRandomSparseNNZ(weak_dim, density, total_nnz_weak);
            std::cout << "    Actual nnz generated: " << weak_global.nnz << std::endl;
            std::cout << "    Actual density: "
                      << (100.0 * weak_global.nnz) / (weak_global.rows * weak_global.cols)
                      << "%" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error generating weak scaling matrix: " << e.what() << std::endl;
            weak_scaling_success = false;
        }
    }

    MPI_Bcast(&weak_scaling_success, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (weak_scaling_success) {
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
                std::cout << "    Max memory per rank: " << max_mem_mb_2d << " MB" << std::endl;
            }
        }

        if (rank == 0) {
            std::cout << "\nWeak scaling results saved to: " << ws_csv << std::endl;
        }
    } else if (rank == 0) {
        std::cout << "Skipping weak scaling due to matrix generation error." << std::endl;
    }

    // ============================================================
    // CLEANUP AND FINALIZATION
    // ============================================================
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK COMPLETE ============= " << std::endl;
        std::cout << "Results saved to:" << std::endl;
        std::cout << "  - Strong scaling: " << csv_file << std::endl;
        if (weak_scaling_success) {
            std::cout << "  - Weak scaling: " << ws_csv << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
