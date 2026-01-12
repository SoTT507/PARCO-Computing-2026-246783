// d_matrix.cpp - FULLY CORRECTED VERSION
#include "d_matrix.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

DistributedMatrix::DistributedMatrix(const COOMatrix& global,
                                     Partitioning part,
                                     MPI_Comm world)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    global_rows = global.rows;
    global_cols = global.cols;

    // Force 2D to 1D if only 1 process
    if (part == Partitioning::TwoD && size == 1) {
        part = Partitioning::OneD;
    }

    // ============================================================
    //                      1D PARTITIONING (Cyclic by rows)
    // ============================================================
    if (part == Partitioning::OneD) {
        comm = world;
        row_comm = MPI_COMM_NULL;
        col_comm = MPI_COMM_NULL;
        Pr = size;
        Pc = 1;

        local_cols = global_cols;

        // Cyclic row distribution: row i goes to process i % size
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

        for (int k = 0; k < global.nnz; ++k) {
            int global_row = global.row_idx[k];
            if (global_to_local[global_row] != -1) {
                local_coo.addEntry(
                    global_to_local[global_row],
                    global.col_idx[k],  // Keep GLOBAL column index
                    global.values[k]
                );
            }
        }

        // Convert to CSR
        local_csr = CSRMatrix(local_coo);

        if (rank == 0) {
            std::cout << "1D Partitioning: " << size << " processes" << std::endl;
            std::cout << "  Local rows per process: ~" << local_rows << std::endl;
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

        // Create local COO with GLOBAL column indices (critical!)
        // With this fix:
        COOMatrix local_coo(local_rows, local_cols);  // Note: local_cols, not global_cols

        int local_nnz_count = 0;
        for (int k = 0; k < global.nnz; ++k) {
          int global_row = global.row_idx[k];
          int global_col = global.col_idx[k];

          // Check if this non-zero belongs to our block
          bool in_row_range = (global_row >= row_start && 
                        global_row < row_start + local_rows);
          bool in_col_range = (global_col >= col_start && 
                        global_col < col_start + local_cols);

          if (in_row_range && in_col_range) {
            int local_row = global_row - row_start;
            // Convert to LOCAL column index
            int local_col = global_col - col_start;
            local_coo.addEntry(local_row, local_col, global.values[k]);
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
