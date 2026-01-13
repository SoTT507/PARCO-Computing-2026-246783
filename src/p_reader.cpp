// simple_parallel_reader.cpp
#include "p_reader.hpp"
#include "d_matrix.hpp"
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
    int rows, cols, nnz;
    read_metadata(filename, rows, cols, nnz, comm);

    // Each process will read the entire file but only keep its rows
    // This is simpler than MPI-IO but still demonstrates parallel reading concept

    COOMatrix local_coo;
    local_coo.rows = 0;  // Will calculate based on our rows
    local_coo.cols = cols;

    // Count how many rows belong to this process
    for (int i = rank; i < rows; i += size) {
        local_coo.rows++;
    }

    // Create mapping for quick lookup
    std::vector<bool> is_my_row(rows, false);
    int local_idx = 0;
    for (int i = rank; i < rows; i += size) {
        is_my_row[i] = true;
    }

    // Now each process reads the file
    // In practice, you'd use MPI-IO here, but for simplicity:
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    // Skip comments and header
    while (std::getline(file, line)) {
        if (line[0] != '%') {
            // First data line (dimensions) - we already have it
            break;
        }
    }

    // Read matrix entries
    int entries_read = 0;
    while (entries_read < nnz && std::getline(file, line)) {
        std::istringstream iss(line);
        int row, col;
        double value;

        if (iss >> row >> col >> value) {
            // Convert to 0-based
            row--;
            col--;

            if (is_my_row[row]) {
                // Calculate local row index
                int local_row = 0;
                for (int r = rank; r < row; r += size) {
                    if (r < rows) local_row++;
                }

                local_coo.addEntry(local_row, col, value);
            }
            entries_read++;
        }
    }

    file.close();
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
