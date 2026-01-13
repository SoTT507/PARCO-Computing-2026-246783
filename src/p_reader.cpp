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
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        // Skip comments
        while (std::getline(file, line)) {
            if (line[0] != '%') break;
        }

        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        file.close();
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, comm);
}

COOMatrix SimpleParallelReader::read_1D_cyclic(const std::string& filename,
                                               int rank, int size,
                                               MPI_Comm comm) {
    int rows = 0, cols = 0, nnz = 0;
    read_metadata(filename, rows, cols, nnz, comm);

    // Count rows for this process
    int local_rows = 0;
    std::vector<int> my_rows;
    for (int i = rank; i < rows; i += size) {
        my_rows.push_back(i);
        local_rows++;
    }

    COOMatrix local_coo(local_rows, cols);
    local_coo.nnz = 0;  // Initialize

    // DEBUG: Print info
    if (rank == 0) {
        std::cout << "DEBUG: Matrix " << rows << "x" << cols << ", nnz=" << nnz << std::endl;
        std::cout << "DEBUG: Process " << rank << " has " << local_rows << " local rows" << std::endl;
    }

    // Create a map from global row to local row index
    std::unordered_map<int, int> global_to_local;
    for (size_t i = 0; i < my_rows.size(); i++) {
        global_to_local[my_rows[i]] = i;
    }

    // Each process reads the entire file
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            // First data line is dimensions - we already have it
            break;
        }
    }

    // Read matrix entries
    int entries_read = 0;
    int entries_kept = 0;
    
    while (entries_read < nnz && std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        int row, col;
        double value;

        if (iss >> row >> col >> value) {
            // Convert from 1-based to 0-based
            row--;
            col--;
            
            // Check if this entry belongs to our process
            auto it = global_to_local.find(row);
            if (it != global_to_local.end()) {
                int local_row = it->second;
                
                // Validate indices before adding
                if (local_row >= 0 && local_row < local_rows && 
                    col >= 0 && col < cols) {
                    local_coo.addEntry(local_row, col, value);
                    entries_kept++;
                } else {
                    // DEBUG: Print error
                    std::cerr << "WARNING: Invalid indices - local_row=" << local_row 
                              << ", col=" << col << std::endl;
                }
            }
            entries_read++;
            
            // Progress indicator
            if (entries_read % 100000 == 0 && rank == 0) {
                std::cout << "DEBUG: Read " << entries_read << "/" << nnz << " entries" << std::endl;
            }
        }
    }

    file.close();
    
    // DEBUG: Print summary
    if (rank == 0) {
        std::cout << "DEBUG: Process " << rank << " kept " << entries_kept 
                  << " entries out of " << nnz << std::endl;
    }

    // Verify we read all entries
    if (entries_read != nnz) {
        std::cerr << "WARNING: Process " << rank << " only read " << entries_read 
                  << "/" << nnz << " entries" << std::endl;
    }

    return local_coo;
}
// In DistributedMatrix::read_local_portion
COOMatrix DistributedMatrix::read_local_portion(const std::string& filename,
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
