// p_reader.hpp
#ifndef SIMPLE_PARALLEL_READER_HPP
#define SIMPLE_PARALLEL_READER_HPP

#include "s_matrix.hpp"
#include "pch.h"
#include <mpi.h>
#include <string>

struct MMHeaderInfo {
    int rows = 0;
    int cols = 0;
    int nnz_header = 0;
    bool is_pattern = false;
    bool is_symmetric = false;
    MPI_Offset data_start = 0; // byte offset where coordinate triplets begin
};

class SimpleParallelReader {
public:
    // Parse MatrixMarket header using mmio on rank 0, broadcast to all ranks
    static MMHeaderInfo read_mmio_header(const std::string& filename, MPI_Comm comm);

    // MPI-IO implementation (MPI_File_read_at_all) for 1D cyclic partitioning
    // Keeps only entries for owner(row)=row%P, stores LOCAL row indices
    static COOMatrix read_1D_cyclic_mpiio(const std::string& filename,
                                          int rank, int size,
                                          MPI_Comm comm);

    // MPI-IO + owner-by-grid redistribution for 2D block partitioning
    // each rank parses a chunk, then sends each entry to its owning block rank
    // output COO uses LOCAL row/col indices within the owned block.
    static COOMatrix read_2D_block_mpiio_redistribute(const std::string& filename,
                                                      int Pr, int Pc,
                                                      int my_r, int my_c,
                                                      MPI_Comm comm);
};

#endif
