#ifndef D_MATRIX_HPP
#define D_MATRIX_HPP

#include "s_matrix.hpp"
#include <mpi.h>
#include <vector>

enum class Partitioning {
    OneD,
    TwoD
};

class DistributedMatrix {
public:
    // MPI info
    int rank;
    int size;
    int omp_num_threads;
    MPI_Comm comm;
    MPI_Comm row_comm;  // For 2D: processes in same row
    MPI_Comm col_comm;  // For 2D: processes in same column

    // Matrix dimensions
    int global_rows, global_cols;
    int local_rows, local_cols;

    std::vector<int>col_block_starts;
    std::vector<size_t>col_block_sizes;

    // 2D grid info
    int dims[2];
    int coords[2];
    int Pr, Pc;          // Grid dimensions
    int my_r, my_c;      // My coordinates
    int row_start, col_start;  // My block starting indices

    // Local CSR matrix
    CSRMatrix local_csr;

    // Communication buffers for 1D
    std::vector<int> recvcounts;
    std::vector<int> displs;

    DistributedMatrix(const COOMatrix& global,
                          Partitioning part,
                          MPI_Comm world = MPI_COMM_WORLD,
                          bool already_distributed = false);

    // Constructor with parallel I/O (MPI-IO in reader)
    DistributedMatrix(const std::string& filename,
                      Partitioning part,
                      MPI_Comm world = MPI_COMM_WORLD);

    static DistributedMatrix FromFileParallel(const std::string& filename,
                                              Partitioning part,
                                              MPI_Comm world = MPI_COMM_WORLD);

    void initialize_partitioning(const COOMatrix& matrix_data, Partitioning part, MPI_Comm world, bool is_already_distributed);

    void spmv(const std::vector<double>& x_global,
              std::vector<double>& y_local,
              double* comm_time_ms = nullptr,
              double* comp_time_ms = nullptr) const;

    void spmv_2d_full(const std::vector<double>& x_global,
                      std::vector<double>& y_local) const;

    void print_partitioning_info() const;
    size_t get_local_nnz() const;
    double get_load_imbalance() const;

    size_t getLocalMemoryUsage() const;

    void printInfo() const;

private:
    DistributedMatrix(const DistributedMatrix&) = delete;
    DistributedMatrix& operator=(const DistributedMatrix&) = delete;

    COOMatrix read_local_portion(const std::string& filename,
                                 Partitioning part,
                                 MPI_Comm world) const;

    void initialize_from_local_coo(const COOMatrix& local_coo,
                                   Partitioning part,
                                   MPI_Comm world);
};

#endif
