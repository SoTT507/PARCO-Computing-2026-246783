#include "d_matrix.hpp"

DistributedMatrix::DistributedMatrix(const COOMatrix& global,
                                     Partitioning part,
                                     MPI_Comm world) {
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    global_rows = global.rows;
    global_cols = global.cols;

    if (part == Partitioning::OneD) {
        // ---------- 1D PARTITIONING ----------
        comm = world;
        local_cols = global_cols;

        COOMatrix local_coo(0, global_cols);

        for (int k = 0; k < global.nnz; ++k) {
            int i = global.row_idx[k];
            if (i % size == rank) {
                local_coo.addEntry(
                    i / size,
                    global.col_idx[k],
                    global.values[k]
                );
            }
        }

        local_rows = (global_rows + size - 1) / size;
        local_csr = CSRMatrix(local_coo);
        row_comm = col_comm = MPI_COMM_NULL;
    }

    else {
        // ---------- 2D PARTITIONING ----------
        dims[0] = dims[1] = 0;
        MPI_Dims_create(size, 2, dims);

        int periods[2] = {0, 0};
        MPI_Cart_create(world, 2, dims, periods, 1, &comm);
        MPI_Cart_coords(comm, rank, 2, coords);

        int Pr = dims[0], Pc = dims[1];
        int my_r = coords[0], my_c = coords[1];

        MPI_Comm_split(comm, my_r, my_c, &row_comm);
        MPI_Comm_split(comm, my_c, my_r, &col_comm);

        int block_rows = (global_rows + Pr - 1) / Pr;
        int block_cols = (global_cols + Pc - 1) / Pc;

        local_rows = block_rows;
        local_cols = block_cols;

        COOMatrix local_coo(block_rows, block_cols);

        for (int k = 0; k < global.nnz; ++k) {
            int i = global.row_idx[k];
            int j = global.col_idx[k];

            int pr = i / block_rows;
            int pc = j / block_cols;

            if (pr == my_r && pc == my_c) {
                local_coo.addEntry(
                    i - pr * block_rows,
                    j - pc * block_cols,
                    global.values[k]
                );
            }
        }

        local_csr = CSRMatrix(local_coo);
    }
}

// ====================== SpMV ======================
void DistributedMatrix::spmv(const std::vector<double>& x_global,
                             std::vector<double>& y_local) const {

    if (row_comm == MPI_COMM_NULL) {
        // ---------- 1D ----------
        std::vector<double> x = x_global;
        local_csr.spmv(x, y_local);
        return;
    }

    // ---------- 2D ----------
    int Pc = dims[1];
    int block_cols = local_cols;

    std::vector<double> x_block(block_cols);
    std::vector<double> y_partial(local_rows, 0.0);

    for (int k = 0; k < Pc; ++k) {
        if (coords[1] == k) {
            std::copy(
                x_global.begin() + k * block_cols,
                x_global.begin() + (k + 1) * block_cols,
                x_block.begin()
            );
        }

        MPI_Bcast(x_block.data(), block_cols, MPI_DOUBLE, k, col_comm);

        #pragma omp parallel for
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i];
                 j < local_csr.row_ptr[i + 1]; ++j) {
                sum += local_csr.values[j] *
                       x_block[local_csr.col_idx[j]];
            }
            y_partial[i] += sum;
        }
    }

    y_local.resize(local_rows);
    MPI_Reduce(y_partial.data(), y_local.data(),
               local_rows, MPI_DOUBLE, MPI_SUM,
               0, row_comm);
}
