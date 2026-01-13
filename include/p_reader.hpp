// simple_parallel_reader.hpp
#ifndef SIMPLE_PARALLEL_READER_HPP
#define SIMPLE_PARALLEL_READER_HPP

#include "s_matrix.hpp"
#include "pch.h"
#include <mpi.h>
#include <string>
#include <vector>
#include <fstream>

class SimpleParallelReader {
public:
    // Read metadata (small, broadcast to all)
    static void read_metadata(const std::string& filename,
                              int& rows, int& cols, int& nnz,
                              MPI_Comm comm);

    // Each process reads its rows based on partitioning
    static COOMatrix read_1D_cyclic(const std::string& filename,
                                    int rank, int size,
                                    MPI_Comm comm);

    static COOMatrix read_2D_block(const std::string& filename,
                                   int Pr, int Pc,
                                   int my_r, int my_c,
                                   MPI_Comm comm);
};

#endif
