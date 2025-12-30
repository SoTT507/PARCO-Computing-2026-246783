#pragma once
#include "pch.h"
#include "d_matrix.hpp"

struct MPITiming {
    double spmv_time_ms;
};

class MPIBenchmark {
public:
    static MPITiming benchmark_spmv(const DistributedMatrix& A, const std::vector<double>& x, int runs);
    static void write_csv_header(const std::string& filename);
    static void write_csv_row(const std::string& filename, const std::string& matrix,
                              const std::string& partitioning, int mpi_procs,
                              int omp_threads, const MPITiming& t);
};
