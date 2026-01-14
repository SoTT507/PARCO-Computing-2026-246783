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
// WEAK SCALING (Random Matrix with uniform NNZ per rank) — distributed synthetic generation
// ============================================================

  MPI_Barrier(MPI_COMM_WORLD);
if (rank == 0) {
    std::cout << "\n ============= WEAK SCALING (Uniform NNZ per rank) =============\n";
}

// Constant workload per rank
const int nnz_per_rank = 100000;

// Controls global growth with P (weak scaling): rows ~ P
const int base_rows_per_rank = 4096;
const int global_rows_ws = base_rows_per_rank * size;
const int global_cols_ws = global_rows_ws; // square

const uint64_t seed = 12345;

std::string ws_csv = "mpi_weak_scaling.csv";
if (rank == 0) {
    SparseMatrixBenchmark::writeMPIcsvHeader(ws_csv);
}

// ------------------------------------------------------------
// Helpers (put these near the top of main.cpp or as lambdas)
// ------------------------------------------------------------
auto gen_local_1d_cyclic = [&](int Grows, int Gcols, int nnz_rank) -> COOMatrix {
    // local rows for cyclic distribution
    int lrows = 0;
    for (int i = rank; i < Grows; i += size) ++lrows;

    COOMatrix local(lrows, Gcols);

    std::mt19937_64 rng(seed + (uint64_t)rank * 1315423911ULL);
    std::uniform_int_distribution<int> rdist(0, std::max(0, lrows - 1));
    std::uniform_int_distribution<int> cdist(0, std::max(0, Gcols - 1));

    // If lrows==0 (can happen only if Grows < size), avoid UB:
    if (lrows == 0 || Gcols == 0) return local;

    for (int k = 0; k < nnz_rank; ++k) {
        int lr = rdist(rng);
        int gc = cdist(rng);
        local.addEntry(lr, gc, 1.0);
    }
    return local;
};

auto gen_local_2d_block = [&](int Grows, int Gcols, int nnz_rank,
                              int Pr, int Pc, int my_r, int my_c) -> COOMatrix {
    // Must match the uneven block scheme used in reader/2D partitioning
    auto block_start_uneven = [](int n, int P, int p) {
        int base = n / P, rem = n % P;
        return p * base + std::min(p, rem);
    };
    auto block_size_uneven = [](int n, int P, int p) {
        int base = n / P, rem = n % P;
        return base + (p < rem ? 1 : 0);
    };

    int lrows = block_size_uneven(Grows, Pr, my_r);
    int lcols = block_size_uneven(Gcols, Pc, my_c);

    COOMatrix local(lrows, lcols);

    std::mt19937_64 rng(seed + (uint64_t)rank * 11400714819323198485ULL);
    std::uniform_int_distribution<int> rdist(0, std::max(0, lrows - 1));
    std::uniform_int_distribution<int> cdist(0, std::max(0, lcols - 1));

    if (lrows == 0 || lcols == 0) return local;

    for (int k = 0; k < nnz_rank; ++k) {
        int lr = rdist(rng);
        int lc = cdist(rng);
        local.addEntry(lr, lc, 1.0);
    }
    return local;
};

// ------------------------------------------------------------
// 1D weak scaling (uniform nnz per rank)
// ------------------------------------------------------------
{
    COOMatrix local_1d = gen_local_1d_cyclic(global_rows_ws, global_cols_ws, nnz_per_rank);

    // IMPORTANT: use the constructor that accepts explicit global dims
    DistributedMatrix A1(local_1d, Partitioning::OneD, MPI_COMM_WORLD,
                         global_rows_ws, global_cols_ws);

    std::vector<double> x(global_cols_ws, 1.0);

    size_t local_mem = A1.getLocalMemoryUsage();
    size_t max_mem = 0;
    MPI_Reduce(&local_mem, &max_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    double mem_mb = max_mem / (1024.0 * 1024.0);

    BenchmarkResult res = SparseMatrixBenchmark::benchmark_spmv(A1, x, 10);

    const long long weak_nnz_total = (long long)nnz_per_rank * (long long)size;

    if (rank == 0) {
        SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "RandomWeakUniform", "1D",
                                              size, omp_threads, weak_nnz_total,
                                              mem_mb, res);
    }
}

// ------------------------------------------------------------
// 2D weak scaling (uniform nnz per rank)
// ------------------------------------------------------------
if (size > 1) {
    int grid[2] = {0, 0};
    MPI_Dims_create(size, 2, grid);
    int Pr = grid[0];
    int Pc = grid[1];
    int my_r = rank / Pc;
    int my_c = rank % Pc;

    COOMatrix local_2d = gen_local_2d_block(global_rows_ws, global_cols_ws, nnz_per_rank,
                                           Pr, Pc, my_r, my_c);

    DistributedMatrix A2(local_2d, Partitioning::TwoD, MPI_COMM_WORLD,
                         global_rows_ws, global_cols_ws);

    std::vector<double> x2(global_cols_ws, 1.0);

    size_t local_mem2 = A2.getLocalMemoryUsage();
    size_t max_mem2 = 0;
    MPI_Reduce(&local_mem2, &max_mem2, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    double mem_mb2 = max_mem2 / (1024.0 * 1024.0);

    BenchmarkResult res2 = SparseMatrixBenchmark::benchmark_spmv(A2, x2, 10);

    const long long weak_nnz_total = (long long)nnz_per_rank * (long long)size;

    if (rank == 0) {
        SparseMatrixBenchmark::writeMPIcsvRow(ws_csv, "RandomWeakUniform", "2D",
                                              size, omp_threads, weak_nnz_total,
                                              mem_mb2, res2);
    }
}

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n ============= BENCHMARK COMPLETE =============\n";
        std::cout << "Strong scaling: " << csv_file << "\n";
        std::cout << "Weak scaling:   " << ws_csv << "\n";
    }

    MPI_Finalize();
    return 0;
}
