// d_matrix.cpp - FULLY CORRECTED VERSION
#include "d_matrix.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

DistributedMatrix::DistributedMatrix(const COOMatrix& global,
                                     Partitioning part,
                                     MPI_Comm world,
                                     bool already_distributed)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

    global_rows = global.rows;
    global_cols = global.cols;

    // Force 2D to 1D if only 1 process
    if (part == Partitioning::TwoD && size == 1) {
        part = Partitioning::OneD;
    }

    if (!already_distributed) {
        // ============================================
        // ORIGINAL CODE: Broadcast from rank 0
        // ============================================
        COOMatrix global_copy = global;  // Work with copy

        // Broadcast dimensions
        MPI_Bcast(&global_copy.rows, 1, MPI_INT, 0, world);
        MPI_Bcast(&global_copy.cols, 1, MPI_INT, 0, world);
        MPI_Bcast(&global_copy.nnz, 1, MPI_INT, 0, world);

        // Allocate on non-root ranks
        if (rank != 0) {
            global_copy.row_idx.resize(global_copy.nnz);
            global_copy.col_idx.resize(global_copy.nnz);
            global_copy.values.resize(global_copy.nnz);
        }

        // Broadcast matrix data
        MPI_Bcast(global_copy.row_idx.data(), global_copy.nnz, MPI_INT, 0, world);
        MPI_Bcast(global_copy.col_idx.data(), global_copy.nnz, MPI_INT, 0, world);
        MPI_Bcast(global_copy.values.data(), global_copy.nnz, MPI_DOUBLE, 0, world);

        // Now continue with partitioning logic using global_copy
        initialize_partitioning(global_copy, part, world, false);
    } else {
        // ============================================
        // NEW: Matrix is already local (no broadcast needed)
        // ============================================
        // 'global' parameter actually contains LOCAL data for this process
        initialize_partitioning(global, part, world, true);
    }
}

void DistributedMatrix::initialize_partitioning(const COOMatrix& matrix_data,
                                                Partitioning part,
                                                MPI_Comm world,
                                                bool is_already_distributed)
{
    // Store global dimensions
    global_rows = matrix_data.rows;
    global_cols = matrix_data.cols;

    // Force 2D to 1D if only 1 process
    if (part == Partitioning::TwoD && size == 1) {
        part = Partitioning::OneD;
    }

    // ============================================================
    //                      1D PARTITIONING (Cyclic by rows)
    // ============================================================
    if (part == Partitioning::OneD) {
        row_comm = MPI_COMM_NULL;
        col_comm = MPI_COMM_NULL;
        Pr = size;
        Pc = 1;

        local_cols = global_cols;

        if (is_already_distributed) {
            // For parallel I/O: matrix_data already contains our local rows
            local_rows = matrix_data.rows;  // Already the local row count

            // Convert local COO to CSR directly
            local_csr = CSRMatrix(matrix_data);

            if (rank == 0) {
                std::cout << "1D Parallel Partitioning: " << size << " processes" << std::endl;
                std::cout << "  Local rows per process: variable" << std::endl;
            }
        } else {
            // Original sequential approach
            local_rows = 0;
            for (int i = rank; i < global_rows; i += size) {
                local_rows++;
            }

            // Create mapping from global row to local row
            std::vector<int> global_to_local(global_rows, -1);
            int local_idx = 0;
            for (int i = rank; i < global_rows; i += size) {
                global_to_local[i] = local_idx++;
            }

            // Create local COO with GLOBAL column indices
            COOMatrix local_coo(local_rows, global_cols);

            for (int k = 0; k < matrix_data.nnz; ++k) {
                int global_row = matrix_data.row_idx[k];
                if (global_to_local[global_row] != -1) {
                    local_coo.addEntry(
                        global_to_local[global_row],
                        matrix_data.col_idx[k],  // Keep GLOBAL column index
                        matrix_data.values[k]
                    );
                }
            }

            // Convert to CSR
            local_csr = CSRMatrix(local_coo);

            if (rank == 0) {
                std::cout << "1D Sequential Partitioning: " << size << " processes" << std::endl;
                std::cout << "  Local rows per process: ~" << local_rows << std::endl;
            }
        }
    }
    // ============================================================
    //                      2D PARTITIONING (Block layout)
    // ============================================================
    else if (part == Partitioning::TwoD) {
        // Determine optimal 2D grid (Pr x Pc)
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);
        Pr = dims[0];
        Pc = dims[1];

        // Create Cartesian communicator
        int periods[2] = {0, 0};
        MPI_Cart_create(world, 2, dims, periods, 1, &comm);
        MPI_Cart_coords(comm, rank, 2, coords);

        my_r = coords[0];
        my_c = coords[1];

        // Create row and column communicators
        MPI_Comm_split(comm, my_r, rank, &row_comm);
        MPI_Comm_split(comm, my_c, rank, &col_comm);

        // Calculate block sizes with load balancing
        int rows_per_proc = global_rows / Pr;
        int extra_rows = global_rows % Pr;

        local_rows = rows_per_proc;
        if (my_r < extra_rows) local_rows++;

        row_start = 0;
        for (int r = 0; r < my_r; r++) {
            int block_rows = rows_per_proc;
            if (r < extra_rows) block_rows++;
            row_start += block_rows;
        }

        int cols_per_proc = global_cols / Pc;
        int extra_cols = global_cols % Pc;

        local_cols = cols_per_proc;
        if (my_c < extra_cols) local_cols++;

        col_start = 0;
        for (int c = 0; c < my_c; c++) {
            int block_cols = cols_per_proc;
            if (c < extra_cols) block_cols++;
            col_start += block_cols;
        }

        // Precompute column block info for all blocks (needed for SpMV)
        col_block_starts.resize(Pc + 1);
        col_block_sizes.resize(Pc);

        col_block_starts[0] = 0;
        for (int c = 0; c < Pc; c++) {
            int block_cols = cols_per_proc;
            if (c < extra_cols) block_cols++;
            col_block_sizes[c] = block_cols;
            col_block_starts[c + 1] = col_block_starts[c] + block_cols;
        }

        if (rank == 0) {
            std::cout << "2D Partitioning: " << Pr << "x" << Pc << " grid" << std::endl;
        }

        if (is_already_distributed) {
            // For parallel I/O: matrix_data already contains our local block
            // Just convert to CSR
            local_csr = CSRMatrix(matrix_data);

            if (rank == 0) {
                std::cout << "  Block sizes: variable per process" << std::endl;
            }
        } else {
            // Original sequential approach
            // Create local COO with LOCAL column indices
            COOMatrix local_coo(local_rows, local_cols);

            int local_nnz_count = 0;
            for (int k = 0; k < matrix_data.nnz; ++k) {
                int global_row = matrix_data.row_idx[k];
                int global_col = matrix_data.col_idx[k];

                // Check if this non-zero belongs to our block
                bool in_row_range = (global_row >= row_start &&
                              global_row < row_start + local_rows);
                bool in_col_range = (global_col >= col_start &&
                              global_col < col_start + local_cols);

                if (in_row_range && in_col_range) {
                    int local_row = global_row - row_start;
                    // Convert to LOCAL column index
                    int local_col = global_col - col_start;
                    local_coo.addEntry(local_row, local_col, matrix_data.values[k]);
                    local_nnz_count++;
                }
            }

            // Convert to CSR
            local_csr = CSRMatrix(local_coo);

            if (rank == 0) {
                std::cout << "  Block sizes: " << local_rows << "x" << local_cols
                          << " per process" << std::endl;
            }
        }
    }
}

DistributedMatrix::DistributedMatrix(const std::string& filename,
                                     Partitioning part,
                                     MPI_Comm world)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

    // Read local portion directly from file
    COOMatrix local_coo = read_local_portion(filename, part, world);

    // Initialize from local data (no broadcast needed)
    initialize_from_local_coo(local_coo, part, world);
}

void DistributedMatrix::initialize_from_local_coo(const COOMatrix& local_coo,
                                                  Partitioning part,
                                                  MPI_Comm world) {
    // First, gather global dimensions from all processes
    int local_dims[2] = {local_coo.rows, local_coo.cols};
    int global_dims[2] = {0, 0};

    // For 1D: rows are distributed, columns are global
    // For 2D: both rows and columns are distributed

    if (part == Partitioning::OneD) {
        // Sum local rows to get global rows
        MPI_Allreduce(&local_coo.rows, &global_rows, 1, MPI_INT, MPI_SUM, world);
        // Columns are the same for all processes
        MPI_Allreduce(&local_coo.cols, &global_cols, 1, MPI_INT, MPI_MAX, world);
    } else {
        // For 2D, we need to know the process grid first
        // ... determine grid and calculate global dimensions
    }

    // Now call the partitioning logic
    initialize_partitioning(local_coo, part, world, true);
}

DistributedMatrix DistributedMatrix::FromFileParallel(const std::string& filename,
                                                      Partitioning part,
                                                      MPI_Comm world) {
    return DistributedMatrix(filename, part, world);
}

// ============================================================
// ================= CORRECTED SpMV ===========================
// ============================================================
void DistributedMatrix::spmv(const std::vector<double>& x_global,
                             std::vector<double>& y_local,
                             double* comm_time_ms,
                             double* comp_time_ms) const
{
    using namespace std::chrono;

    // Initialize timing
    if (comm_time_ms) *comm_time_ms = 0.0;
    if (comp_time_ms) *comp_time_ms = 0.0;

    // ============================================================
    //                      1D PARTITIONING
    // ============================================================
    if (row_comm == MPI_COMM_NULL) {
        y_local.resize(local_rows);

        // NO COMMUNICATION NEEDED - x_global already replicated on all ranks
        // (This is the O(P·m) memory overhead mentioned in the paper)

        auto comp_start = high_resolution_clock::now();

        // Simple computation - all processes have full x vector
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int global_col = local_csr.col_idx[j];
                sum += local_csr.values[j] * x_global[global_col];
            }
            y_local[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();

        if (comp_time_ms) {
            *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        }
        // comm_time_ms stays 0 for 1D
    }
    // ============================================================
    //                      2D PARTITIONING (Pc == 1)
    // ============================================================
    else if (Pc == 1) {
        // This is effectively 1D row distribution using 2D infrastructure
        y_local.resize(local_rows);

        auto comp_start = high_resolution_clock::now();


        // #pragma omp parallel num_threads(OMP_NUM_THREADS)
        // {
          #pragma omp for schedule(guided)
          for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int global_col = local_csr.col_idx[j];
                sum += local_csr.values[j] * x_global[global_col];
            }
            y_local[i] = sum;
          }
        // }

        auto comp_end = high_resolution_clock::now();

        if (comp_time_ms) {
            *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        }
    }
    // ============================================================
    //                 TRUE 2D PARTITIONING (Pr > 1 && Pc > 1)
    // ============================================================
    else {
        // PHASE 1: BROADCAST X BLOCKS ALONG COLUMNS
        // Each column group gets its x_block from the root row
        std::vector<double> x_block(local_cols, 0.0);

        auto comm1_start = high_resolution_clock::now();

        // Process at row 0 in each column extracts its x block
        if (my_r == 0) {
            if (col_start < (int)x_global.size()) {
                int copy_size = std::min(local_cols, (int)x_global.size() - col_start);
                std::copy(x_global.begin() + col_start,
                         x_global.begin() + col_start + copy_size,
                         x_block.begin());
            }
        }

        // Broadcast x_block to all processes in this column
        MPI_Bcast(x_block.data(), local_cols, MPI_DOUBLE, 0, col_comm);

        auto comm1_end = high_resolution_clock::now();

        // PHASE 2: LOCAL COMPUTATION
        // Each process computes partial results using its local block and x_block
        std::vector<double> y_partial(local_rows, 0.0);

        auto comp_start = high_resolution_clock::now();

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
          double sum = 0.0;
          for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
            int local_col = local_csr.col_idx[j];  // Already local after fix
            sum += local_csr.values[j] * x_block[local_col];
          }
          y_partial[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();

        // PHASE 3: REDUCE RESULTS ALONG ROWS
        // Sum partial results from all column blocks to get final y
        auto comm2_start = high_resolution_clock::now();

        y_local.resize(local_rows, 0.0);

        // Reduce-scatter pattern: sum across row communicator
        // Result is distributed - each column group has valid result
        MPI_Allreduce(y_partial.data(), y_local.data(), local_rows,
                      MPI_DOUBLE, MPI_SUM, row_comm);

        auto comm2_end = high_resolution_clock::now();

        // Calculate timing
        if (comm_time_ms) {
            *comm_time_ms = duration<double, std::milli>(comm1_end - comm1_start).count() +
                           duration<double, std::milli>(comm2_end - comm2_start).count();
        }
        if (comp_time_ms) {
            *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        }
    }
}// ============================================================
// ================= HELPER FUNCTIONS =========================
// ============================================================
size_t DistributedMatrix::getLocalMemoryUsage() const {
    size_t mem = 0;
    // CSR arrays
    mem += local_csr.values.capacity() * sizeof(double);
    mem += local_csr.col_idx.capacity() * sizeof(int);
    mem += local_csr.row_ptr.capacity() * sizeof(int);
    return mem;
}

void DistributedMatrix::printInfo() const {
    std::cout << "Rank " << rank << ": "
              << "Global " << global_rows << "x" << global_cols
              << ", Local " << local_rows << "x" << local_cols
              << ", CSR nnz=" << local_csr.nnz;
    if (row_comm != MPI_COMM_NULL) {
        std::cout << " (2D " << Pr << "x" << Pc << ")";
    } else {
        std::cout << " (1D)";
    }
    std::cout << std::endl;
}

double DistributedMatrix::get_load_imbalance() const {
    size_t local_nnz = local_csr.nnz;
    size_t max_nnz, min_nnz, total_nnz;

    MPI_Allreduce(&local_nnz, &max_nnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&local_nnz, &min_nnz, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    double avg_nnz = total_nnz / (double)size;
    if (avg_nnz == 0) return 0.0;
    return (max_nnz - min_nnz) / avg_nnz;
}

size_t DistributedMatrix::get_local_nnz() const {
    return local_csr.nnz;
}
