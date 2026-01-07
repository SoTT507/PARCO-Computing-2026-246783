#pragma once
#include "pch.h"
#include "s_matrix.hpp"

enum class Partitioning{
  OneD,
  TwoD
};

class DistributedMatrix{
public:
  CSRMatrix local_csr;

  int global_rows, global_cols;
  int local_rows, local_cols;

  MPI_Comm comm;
  MPI_Comm row_comm, col_comm;

  int rank, size;
  int dims[2], coords[2];

  DistributedMatrix(const COOMatrix& global, Partitioning par, MPI_Comm world = MPI_COMM_WORLD);

  void spmv(const std::vector<double>& x_global, std::vector<double>& y_local) const;
};
