// d_matrix.cpp - CORRECTED VERSION
#include "d_matrix.hpp"
#include <iostream>
#include <algorithm>

DistributedMatrix::DistributedMatrix(const COOMatrix& global,
                                     Partitioning part,
                                     MPI_Comm world)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    global_rows = global.rows;
    global_cols = global.cols;

    // ============================================================
    //                      1D PARTITIONING (Cyclic by rows)
    // ============================================================
    if (part == Partitioning::OneD) {
        comm = world;
        row_comm = MPI_COMM_NULL;
        col_comm = MPI_COMM_NULL;

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

        // Create local COO
        COOMatrix local_coo(local_rows, global_cols);

        for (int k = 0; k < global.nnz; ++k) {
            int global_row = global.row_idx[k];
            if (global_to_local[global_row] != -1) {
                local_coo.addEntry(
                    global_to_local[global_row],
                    global.col_idx[k],
                    global.values[k]
                );
            }
        }

        // Convert to CSR
        local_csr = CSRMatrix(local_coo);

        if (rank == 0) {
            std::cout << "1D Partitioning: " << size << " processes" << std::endl;
            std::cout << "  Each process has ~" << local_rows << " rows" << std::endl;
        }
    }
    // ============================================================
    //                      2D PARTITIONING (Block layout)
    // ============================================================
    else if (part == Partitioning::TwoD) {
            std::cout << "Rank " << rank << ": Starting 2D partitioning" << std::endl;

            // Determine optimal 2D grid (Pr x Pc)
            dims[0] = dims[1] = 0;
            MPI_Dims_create(size, 2, dims);
            Pr = dims[0];  // Number of rows in process grid
            Pc = dims[1];  // Number of columns in process grid

            std::cout << "Rank " << rank << ": Grid " << Pr << "x" << Pc << std::endl;

            // Create Cartesian communicator
            int periods[2] = {0, 0};
            MPI_Cart_create(world, 2, dims, periods, 1, &comm);
            MPI_Cart_coords(comm, rank, 2, coords);

            my_r = coords[0];
            my_c = coords[1];

            std::cout << "Rank " << rank << ": Coordinates (" << my_r << "," << my_c << ")" << std::endl;

            MPI_Comm_split(comm, my_r, rank, &row_comm);

            // Always create col_comm (even if Pc=1, it will be a communicator with 1 process)
            MPI_Comm_split(comm, my_c, rank, &col_comm);

            // Debug output
            int row_comm_size, col_comm_size;
            MPI_Comm_size(row_comm, &row_comm_size);
            MPI_Comm_size(col_comm, &col_comm_size);
            std::cout << "Rank " << rank << " (" << my_r << "," << my_c << "): "
                      << "row_comm size=" << row_comm_size
                      << ", col_comm size=" << col_comm_size << std::endl;

            // ========== CALCULATE BLOCK SIZES ==========
            // Row blocks
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

            // Column blocks
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

            std::cout << "Rank " << rank << ": Block " << row_start << ":"
                      << (row_start + local_rows - 1) << " x " << col_start << ":"
                      << (col_start + local_cols - 1) << " (" << local_rows
                      << "x" << local_cols << ")" << std::endl;

            // Validate dimensions
            if (local_rows <= 0 || local_cols <= 0) {
                std::cerr << "Rank " << rank << ": ERROR - Invalid local dimensions!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Create local COO matrix
            COOMatrix local_coo(local_rows, local_cols);

            // CRITICAL FIX: Check if global matrix data is valid before accessing
            if (global.row_idx.size() != (size_t)global.nnz ||
                global.col_idx.size() != (size_t)global.nnz ||
                global.values.size() != (size_t)global.nnz) {
                std::cerr << "Rank " << rank << ": ERROR - Global matrix data inconsistent!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Distribute non-zeros to local COO
            int local_nnz_count = 0;

            for (int k = 0; k < global.nnz; ++k) {
                // SAFETY CHECK: Validate indices before accessing
                int global_row = global.row_idx[k];
                int global_col = global.col_idx[k];

                // Check if indices are valid
                if (global_row < 0 || global_row >= global_rows ||
                    global_col < 0 || global_col >= global_cols) {
                    std::cerr << "Rank " << rank << ": WARNING - Invalid global indices at k=" << k
                              << ": (" << global_row << "," << global_col << ")" << std::endl;
                    continue;
                }

                // Check if this non-zero belongs to our block
                bool in_row_range = (global_row >= row_start && global_row < row_start + local_rows);
                bool in_col_range = (global_col >= col_start && global_col < col_start + local_cols);

                if (in_row_range && in_col_range) {
                    int local_row = global_row - row_start;
                    int local_col = global_col - col_start;

                    // Validate local indices
                    if (local_row < 0 || local_row >= local_rows ||
                        local_col < 0 || local_col >= local_cols) {
                        std::cerr << "Rank " << rank << ": ERROR - Local index calculation failed!" << std::endl;
                        continue;
                    }

                    local_coo.addEntry(local_row, local_col, global.values[k]);
                    local_nnz_count++;
                }
            }

            std::cout << "Rank " << rank << ": Collected " << local_nnz_count
                      << " non-zeros" << std::endl;

            // Convert to CSR
            try {
                local_csr = CSRMatrix(local_coo);
                std::cout << "Rank " << rank << ": CSR created: "
                          << local_csr.rows << "x" << local_csr.cols
                          << " nnz=" << local_csr.nnz << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << ": CSR creation failed: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::cout << "Rank " << rank << ": 2D constructor completed" << std::endl;
            std::cout.flush();
        }
}

// ============================================================
// ================= FIXED 2D SpMV ============================
// ============================================================
// d_matrix_final.cpp - FIXED SPMV FOR TRUE 2D
void DistributedMatrix::spmv(const std::vector<double>& x_global,
                             std::vector<double>& y_local,
                             double* comm_time_ms,
                             double* comp_time_ms) const
{
    using namespace std::chrono;

    // Initialize timing
    if (comm_time_ms) *comm_time_ms = 0.0;
    if (comp_time_ms) *comp_time_ms = 0.0;

    auto total_start = high_resolution_clock::now();

    // ============================================================
    //                      1D PARTITIONING or 2D with Pc=1
    // ============================================================
    if (row_comm == MPI_COMM_NULL || Pc == 1) {
        // This handles both:
        // 1. Actual 1D partitioning
        // 2. 2D partitioning with only 1 column (which is essentially 1D)

        y_local.resize(local_rows);

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
    }
    // ============================================================
    //                      TRUE 2D PARTITIONING (Pr > 1 && Pc > 1)
    // ============================================================
    else {
        // For true 2D: Pr > 1 && Pc > 1

        // SAFETY CHECK: Ensure communicators are valid
        if (col_comm == MPI_COMM_NULL || row_comm == MPI_COMM_NULL) {
            std::cerr << "Rank " << rank << " (" << my_r << "," << my_c << "): "
                      << "ERROR - Missing communicators for true 2D!" << std::endl;
            y_local.clear();
            return;
        }

        // ========== PHASE 1: BROADCAST X BLOCKS ALONG COLUMNS ==========
        std::vector<double> x_block(local_cols, 0.0);

        auto comm1_start = high_resolution_clock::now();

        // Get rank in column communicator
        int col_rank;
        MPI_Comm_rank(col_comm, &col_rank);

        // Process at row 0 in each column gets the x values
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

        // ========== PHASE 2: LOCAL COMPUTATION ==========
        std::vector<double> y_partial(local_rows, 0.0);

        auto comp_start = high_resolution_clock::now();

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int col = local_csr.col_idx[j];
                if (col >= 0 && col < local_cols) {
                    sum += local_csr.values[j] * x_block[col];
                }
            }
            y_partial[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();

        // ========== PHASE 3: REDUCE RESULTS ALONG ROWS ==========
        auto comm2_start = high_resolution_clock::now();

        y_local.resize(local_rows);

        // Get rank in row communicator
        int row_rank;
        MPI_Comm_rank(row_comm, &row_rank);

        // Root for reduction is process at column 0 in this row
        int root_in_row = 0;

        // Allocate buffer for the reduced result
        std::vector<double> reduced_result(local_rows, 0.0);

        // Reduce partial results - only process in column 0 gets the full result
        MPI_Reduce(y_partial.data(), reduced_result.data(), local_rows,
                   MPI_DOUBLE, MPI_SUM, root_in_row, row_comm);

        auto comm2_end = high_resolution_clock::now();

        // ========== PHASE 4: GATHER RESULTS TO ALL PROCESSES ==========
        auto comm3_start = high_resolution_clock::now();

        // Broadcast the reduced result from column 0 to all processes in the row
        MPI_Bcast(reduced_result.data(), local_rows, MPI_DOUBLE, root_in_row, row_comm);

        // All processes in the row now have the complete result for their rows
        y_local = reduced_result;

        auto comm3_end = high_resolution_clock::now();

        // Calculate timing
        if (comm_time_ms) {
            *comm_time_ms = duration<double, std::milli>(comm1_end - comm1_start).count() +
                           duration<double, std::milli>(comm2_end - comm2_start).count() +
                           duration<double, std::milli>(comm3_end - comm3_start).count();
        }
        if (comp_time_ms) {
            *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        }
    }

    auto total_end = high_resolution_clock::now();
    double total_time = duration<double, std::milli>(total_end - total_start).count();

    // Debug output
    if (rank == 0) {
        static int call_count = 0;
        call_count++;
        if (call_count <= 3) {
            std::cout << "SpMV " << call_count << ": ";
            if (row_comm == MPI_COMM_NULL) {
                std::cout << "1D";
            } else if (Pc == 1) {
                std::cout << "2D(Pc=1)";
            } else {
                std::cout << "2D(" << Pr << "x" << Pc << ")";
            }
            std::cout << " - y_local size: " << y_local.size()
                      << ", time: " << total_time << " ms" << std::endl;
        }
    }
}

// ============================================================
// ================= SIMPLIFIED 2D for Pc=1 ===================
// ============================================================
void DistributedMatrix::spmv_2d_full(const std::vector<double>& x_global,
                                     std::vector<double>& y_local) const
{
    // Simplified version for 2D with Pc=1 or fallback to 1D

    if (row_comm == MPI_COMM_NULL || Pc == 1) {
        // Use the main spmv function
        spmv(x_global, y_local);
        return;
    }

    // For true 2D (Pc > 1), implement a simpler version

    // Get x block for our column
    std::vector<double> x_block(local_cols, 0.0);

    if (my_r == 0) {  // Process at row 0 in our column
        if (col_start < (int)x_global.size()) {
            int copy_size = std::min(local_cols, (int)x_global.size() - col_start);
            std::copy(x_global.begin() + col_start,
                     x_global.begin() + col_start + copy_size,
                     x_block.begin());
        }
    }

    // Broadcast to column (if Pc > 1, we have col_comm)
    if (col_comm != MPI_COMM_NULL) {
        MPI_Bcast(x_block.data(), local_cols, MPI_DOUBLE, 0, col_comm);
    }

    // Local computation
    y_local.resize(local_rows);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
            int col = local_csr.col_idx[j];
            if (col >= 0 && col < local_cols) {
                sum += local_csr.values[j] * x_block[col];
            }
        }
        y_local[i] = sum;
    }

    // Reduce along rows if we have multiple columns
    if (row_comm != MPI_COMM_NULL && my_c != 0) {
        // Send our result to process in column 0
        MPI_Send(y_local.data(), local_rows, MPI_DOUBLE,
                rank - my_c, 0, row_comm);
        y_local.clear();  // We don't need the result
    } else if (row_comm != MPI_COMM_NULL && my_c == 0) {
        // Receive and sum results from other columns in this row
        for (int src_col = 1; src_col < Pc; src_col++) {
            std::vector<double> temp(local_rows);
            MPI_Recv(temp.data(), local_rows, MPI_DOUBLE,
                    rank + src_col, 0, row_comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < local_rows; ++i) {
                y_local[i] += temp[i];
            }
        }
    }
}

// ============================================================
// ================= HELPER FUNCTIONS =========================
// ============================================================
void DistributedMatrix::print_partitioning_info() const {
    if (row_comm == MPI_COMM_NULL) {
        std::cout << "Rank " << rank << " (1D): rows " << local_rows
                  << "/" << global_rows << " cols " << local_cols
                  << "/" << global_cols << std::endl;
    } else {
        std::cout << "Rank " << rank << " (" << my_r << "," << my_c << "): "
                  << "block [" << row_start << "-" << row_start + local_rows - 1
                  << "] x [" << col_start << "-" << col_start + local_cols - 1
                  << "] (" << local_rows << "x" << local_cols << ")"
                  << " nnz=" << local_csr.nnz;
        if (Pc == 1) {
            std::cout << " (effectively 1D)";
        }
        std::cout << std::endl;
    }
}

size_t DistributedMatrix::get_local_nnz() const {
    return local_csr.nnz;
}

double DistributedMatrix::get_load_imbalance() const {
    // Calculate load imbalance factor
    size_t local_nnz = local_csr.nnz;
    size_t max_nnz, min_nnz, total_nnz;

    MPI_Allreduce(&local_nnz, &max_nnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&local_nnz, &min_nnz, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    double avg_nnz = total_nnz / (double)size;
    if (avg_nnz == 0) return 0.0;
    return (max_nnz - min_nnz) / avg_nnz;
}

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
              << ", CSR " << local_csr.rows << "x" << local_csr.cols
              << " nnz=" << local_csr.nnz;
    if (row_comm != MPI_COMM_NULL) {
        std::cout << " (2D " << Pr << "x" << Pc << ")";
    }
    std::cout << std::endl;
}
