// simple_parallel_reader.cpp
#include "p_reader.hpp"
#include "d_matrix.hpp"
#include <unordered_map>
#include <sstream>
#include <algorithm>

void SimpleParallelReader::read_metadata(const std::string& filename,
                                         int& rows, int& cols, int& nnz,
                                         MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Use COOMatrix to read metadata (which uses mmio internally)
        COOMatrix temp;
        temp.readMatrixMarket(filename);
        
        rows = temp.rows;
        cols = temp.cols;
        nnz = temp.nnz;
        
        std::cout << "METADATA (via COOMatrix): " << rows << "x" << cols 
                  << ", nnz=" << nnz << std::endl;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, comm);
}

COOMatrix SimpleParallelReader::read_1D_cyclic(const std::string& filename,
                                               int rank, int size,
                                               MPI_Comm comm) {
    // Step 1: Get metadata
    int rows = 0, cols = 0, nnz = 0;
    read_metadata(filename, rows, cols, nnz, comm);
    
    // Step 2: Read full matrix on rank 0 using COOMatrix (which uses mmio)
    COOMatrix global_coo;
    if (rank == 0) {
        std::cout << "Rank 0 reading matrix with COOMatrix::readMatrixMarket()..." << std::endl;
        global_coo.readMatrixMarket(filename);
        std::cout << "Rank 0 read complete: actual nnz=" << global_coo.nnz << std::endl;
        
        // Update nnz with actual value (after symmetric expansion)
        nnz = global_coo.nnz;
    }
    
    // Step 3: Broadcast the actual nnz (after symmetric expansion)
    MPI_Bcast(&nnz, 1, MPI_INT, 0, comm);
    
    // Step 4: Resize arrays on non-root ranks
    if (rank != 0) {
        global_coo.row_idx.resize(nnz);
        global_coo.col_idx.resize(nnz);
        global_coo.values.resize(nnz);
    }
    
    // Step 5: Broadcast the matrix data
    MPI_Bcast(global_coo.row_idx.data(), nnz, MPI_INT, 0, comm);
    MPI_Bcast(global_coo.col_idx.data(), nnz, MPI_INT, 0, comm);
    MPI_Bcast(global_coo.values.data(), nnz, MPI_DOUBLE, 0, comm);
    
    // Set dimensions on all ranks
    global_coo.rows = rows;
    global_coo.cols = cols;
    global_coo.nnz = nnz;
    
    // Step 6: Calculate which rows belong to this process (cyclic distribution)
    int local_rows = 0;
    std::vector<int> my_rows;
    for (int i = rank; i < rows; i += size) {
        my_rows.push_back(i);
        local_rows++;
    }
    
    if (rank == 0) {
        std::cout << "Distributing " << rows << " rows across " << size 
                  << " processes (cyclic)" << std::endl;
    }
    
    // Step 7: Create local COO matrix
    COOMatrix local_coo(local_rows, cols);
    
    // Create mapping from global row to local row
    std::unordered_map<int, int> global_to_local;
    for (size_t i = 0; i < my_rows.size(); i++) {
        global_to_local[my_rows[i]] = static_cast<int>(i);
    }
    
    // Step 8: Filter entries that belong to this process
    int entries_kept = 0;
    for (int k = 0; k < nnz; ++k) {
        int global_row = global_coo.row_idx[k];
        auto it = global_to_local.find(global_row);
        if (it != global_to_local.end()) {
            local_coo.addEntry(it->second, global_coo.col_idx[k], global_coo.values[k]);
            entries_kept++;
        }
    }
    
    local_coo.nnz = entries_kept;
    
    if (rank == 0) {
        std::cout << "Parallel distribution complete." << std::endl;
    }
    
    return local_coo;
}

// For 2D, we'll use the same approach for now
COOMatrix SimpleParallelReader::read_2D_block(const std::string& filename,
                                              int Pr, int Pc,
                                              int my_r, int my_c,
                                              MPI_Comm comm) {
    // For now, use 1D reading as fallback
    int size;
    MPI_Comm_size(comm, &size);
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    return read_1D_cyclic(filename, rank, size, comm);
}COOMatrix DistributedMatrix::read_local_portion(const std::string& filename,
                                                Partitioning part,
                                                MPI_Comm world) const {
    if (part == Partitioning::OneD) {
        return SimpleParallelReader::read_1D_cyclic(filename, rank, size, world);
    } else {
        // For 2D, we need to determine grid first
        // Simplified version - you can expand this
        // For now, fall back to 1D reading
        return SimpleParallelReader::read_1D_cyclic(filename, rank, size, world);
    }
}
