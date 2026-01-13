// p_reader.cpp
// Baseline (required by Deliverable 2):
//   - Rank 0 reads the full Matrix Market file
//   - Rank 0 distributes (NOT broadcasts) the nonzeros to the owner ranks
//     according to 1D cyclic row ownership: owner(i) = i mod P.

#include "p_reader.hpp"
#include <sstream>
#include <algorithm>
#include <numeric>

void SimpleParallelReader::read_metadata(const std::string& filename,
                                         int& rows, int& cols, int& nnz,
                                         MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Lightweight MatrixMarket metadata parse (no full matrix load).
        // We skip comment lines starting with '%', then read: rows cols nnz.
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open MatrixMarket file: " + filename);
        }
        std::string line;
        // header
        if (!std::getline(in, line)) {
            throw std::runtime_error("Empty MatrixMarket file: " + filename);
        }
        // size line
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] == '%') continue;
            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nnz)) {
                throw std::runtime_error("Failed to parse MatrixMarket size line: " + line);
            }
            break;
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, comm);
}

COOMatrix SimpleParallelReader::read_1D_cyclic(const std::string& filename,
                                               int rank,
                                               int size,
                                               MPI_Comm comm)
{
    // Metadata (rows, cols, header-nnz). Note: the real nnz can change after
    // symmetric expansion performed by COOMatrix::readMatrixMarket().
    int rows = 0, cols = 0, nnz_header = 0;
    read_metadata(filename, rows, cols, nnz_header, comm);

    COOMatrix global_coo;
    int nnz_total = 0;
    if (rank == 0) {
        global_coo.readMatrixMarket(filename);
        nnz_total = global_coo.nnz;
    }
    MPI_Bcast(&nnz_total, 1, MPI_INT, 0, comm);

    // Rank 0 buckets nonzeros by destination owner(row)=row%P.
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<int> flat_r, flat_c;
    std::vector<double> flat_v;

    if (rank == 0) {
        std::vector<std::vector<int>> br(size), bc(size);
        std::vector<std::vector<double>> bv(size);
        br.reserve(size); bc.reserve(size); bv.reserve(size);

        for (int k = 0; k < nnz_total; ++k) {
            const int gr = global_coo.row_idx[k];
            const int owner = gr % size;
            br[owner].push_back(gr);
            bc[owner].push_back(global_coo.col_idx[k]);
            bv[owner].push_back(global_coo.values[k]);
        }

        for (int p = 0; p < size; ++p) {
            sendcounts[p] = static_cast<int>(br[p].size());
        }
        std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, displs.begin() + 1);

        const int total = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
        flat_r.resize(total);
        flat_c.resize(total);
        flat_v.resize(total);

        for (int p = 0; p < size; ++p) {
            const int off = displs[p];
            std::copy(br[p].begin(), br[p].end(), flat_r.begin() + off);
            std::copy(bc[p].begin(), bc[p].end(), flat_c.begin() + off);
            std::copy(bv[p].begin(), bv[p].end(), flat_v.begin() + off);
        }
    }

    int local_nnz = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, comm);

    std::vector<int> r_local(local_nnz), c_local(local_nnz);
    std::vector<double> v_local(local_nnz);

    MPI_Scatterv(rank == 0 ? flat_r.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
                 r_local.data(), local_nnz, MPI_INT, 0, comm);
    MPI_Scatterv(rank == 0 ? flat_c.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
                 c_local.data(), local_nnz, MPI_INT, 0, comm);
    MPI_Scatterv(rank == 0 ? flat_v.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 v_local.data(), local_nnz, MPI_DOUBLE, 0, comm);

    // Build local COO with local row indexing (cyclic): local_row = (global_row - rank)/P.
    int local_rows = 0;
    for (int i = rank; i < rows; i += size) local_rows++;

    COOMatrix local_coo(local_rows, cols);
    for (int k = 0; k < local_nnz; ++k) {
        const int gr = r_local[k];
        const int lr = (gr - rank) / size;
        local_coo.addEntry(lr, c_local[k], v_local[k]);
    }
    local_coo.nnz = local_nnz;
    return local_coo;
}

// 2D reader is not implemented here (bonus). Keep 1D fallback.
COOMatrix SimpleParallelReader::read_2D_block(const std::string& filename,
                                              int Pr, int Pc,
                                              int my_r, int my_c,
                                              MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    int rank;
    MPI_Comm_rank(comm, &rank);

    return read_1D_cyclic(filename, rank, size, comm);
}

COOMatrix DistributedMatrix::read_local_portion(const std::string& filename,
                                                Partitioning part,
                                                MPI_Comm world) const
{
    if (part == Partitioning::OneD || size == 1) {
        return SimpleParallelReader::read_1D_cyclic(filename, rank, size, world);
    }
    return SimpleParallelReader::read_2D_block(filename, 0, 0, 0, 0, world);
}
