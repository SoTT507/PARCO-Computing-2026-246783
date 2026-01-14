#include "p_reader.hpp"
#include "mmio.h"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

// -----------------------------
// Helpers: uneven block ownership (needed for 2D owner-by-grid)
// -----------------------------
static inline int owner_block_uneven(int n, int P, int idx) {
    const int base = n / P;
    const int rem  = n % P;
    const int cut  = (base + 1) * rem;
    if (idx < cut) return idx / (base + 1);
    return rem + (idx - cut) / base;
}

static inline int block_start_uneven(int n, int P, int p) {
    const int base = n / P;
    const int rem  = n % P;
    return p * base + std::min(p, rem);
}

static inline int block_size_uneven(int n, int P, int p) {
    const int base = n / P;
    const int rem  = n % P;
    return base + (p < rem ? 1 : 0);
}

static inline const char* skip_spaces(const char* p, const char* end) {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
    return p;
}

static inline bool parse_triplet_line(const char* line, const char* end,
                                      bool is_pattern,
                                      int& r, int& c, double& v)
{
    line = skip_spaces(line, end);
    if (line >= end) return false;
    if (*line == '%') return false;

    char* next = nullptr;

    long rr = std::strtol(line, &next, 10);
    if (next == line) return false;
    line = next;

    long cc = std::strtol(line, &next, 10);
    if (next == line) return false;
    line = next;

    if (is_pattern) {
        v = 1.0;
    } else {
        v = std::strtod(line, &next);
        if (next == line) return false;
    }

    r = static_cast<int>(rr);
    c = static_cast<int>(cc);
    return true;
}

// ------------------------------------------------------------
// Header parse (mmio) + broadcast
// ------------------------------------------------------------
MMHeaderInfo SimpleParallelReader::read_mmio_header(const std::string& filename, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    MMHeaderInfo h;
    int flags[2] = {0, 0}; // pattern, symmetric

    if (rank == 0) {
        FILE* f = std::fopen(filename.c_str(), "r");
        if (!f) throw std::runtime_error("Cannot open MatrixMarket file: " + filename);

        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0) {
            std::fclose(f);
            throw std::runtime_error("mm_read_banner failed for: " + filename);
        }
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
            std::fclose(f);
            throw std::runtime_error("Only sparse MatrixMarket matrices supported: " + filename);
        }

        if (mm_read_mtx_crd_size(f, &h.rows, &h.cols, &h.nnz_header) != 0) {
            std::fclose(f);
            throw std::runtime_error("mm_read_mtx_crd_size failed for: " + filename);
        }

        h.is_pattern   = mm_is_pattern(matcode);
        h.is_symmetric = mm_is_symmetric(matcode);

        // Move to the end of the current line after the size triple.
        int ch;
        while ((ch = std::fgetc(f)) != '\n' && ch != EOF) {}

        long pos = std::ftell(f);
        if (pos < 0) {
            std::fclose(f);
            throw std::runtime_error("ftell failed while computing data_start");
        }
        h.data_start = static_cast<MPI_Offset>(pos);

        std::fclose(f);

        flags[0] = h.is_pattern ? 1 : 0;
        flags[1] = h.is_symmetric ? 1 : 0;
    }

    MPI_Bcast(&h.rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&h.cols, 1, MPI_INT, 0, comm);
    MPI_Bcast(&h.nnz_header, 1, MPI_INT, 0, comm);
    MPI_Bcast(flags, 2, MPI_INT, 0, comm);
    MPI_Bcast(&h.data_start, 1, MPI_OFFSET, 0, comm);

    h.is_pattern   = flags[0] != 0;
    h.is_symmetric = flags[1] != 0;
    return h;
}

// ------------------------------------------------------------
// MPI-IO chunk reader (line-safe) for 1D cyclic ownership
// owner(row)=row%P; local row index lr=(gr-rank)/P
// ------------------------------------------------------------
COOMatrix SimpleParallelReader::read_1D_cyclic_mpiio(const std::string& filename,
                                                    int rank, int size,
                                                    MPI_Comm comm)
{
    MMHeaderInfo h = read_mmio_header(filename, comm);

    MPI_File fh;
    if (MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_File_open failed for: " + filename);
    }

    MPI_Offset file_size = 0;
    MPI_File_get_size(fh, &file_size);
    if (h.data_start >= file_size) {
        MPI_File_close(&fh);
        throw std::runtime_error("Invalid data_start >= file_size for: " + filename);
    }

    MPI_Offset data_bytes = file_size - h.data_start;

    MPI_Offset chunk = data_bytes / size;
    MPI_Offset start = h.data_start + (MPI_Offset)rank * chunk;
    MPI_Offset end   = (rank == size - 1) ? file_size
                                          : (h.data_start + (MPI_Offset)(rank + 1) * chunk);

    // overlap so we can safely scan to newline
    const MPI_Offset overlap = (MPI_Offset)(1 << 20); 
    MPI_Offset read_start = (rank == 0) ? start : std::max(h.data_start, start - overlap);
    MPI_Offset read_end   = (rank == size - 1) ? end   : std::min(file_size, end + overlap);
    MPI_Offset read_nbytes = read_end - read_start;
    if (read_nbytes < 0) read_nbytes = 0;

    std::vector<char> buf((size_t)read_nbytes + 1, '\0');

    MPI_Status st;
    if (MPI_File_read_at_all(fh, read_start, buf.data(), (int)read_nbytes, MPI_BYTE, &st) != MPI_SUCCESS) {
        MPI_File_close(&fh);
        throw std::runtime_error("MPI_File_read_at_all failed for: " + filename);
    }
    MPI_File_close(&fh);

    const char* b = buf.data();
    const char* b_end = b + read_nbytes;

    const char* parse_begin = b + (start - read_start);
    const char* parse_end   = b + (end   - read_start);

    // Align begin: for rank>0 skip partial first line
    if (rank != 0) {
        const char* p = parse_begin;
        while (p < b_end && *p != '\n') ++p;
        if (p < b_end) ++p;
        parse_begin = p;
    }

    // Align end: for non-last extend to newline end to include boundary-crossing line
    if (rank != size - 1) {
        const char* p = parse_end;
        while (p < b_end && *p != '\n') ++p;
        if (p < b_end) ++p;
        parse_end = p;
    } else {
        parse_end = b_end;
    }

    if (parse_begin < b) parse_begin = b;
    if (parse_end > b_end) parse_end = b_end;
    if (parse_begin > parse_end) parse_begin = parse_end;

    // local rows count for cyclic distribution
    int local_rows = 0;
    for (int i = rank; i < h.rows; i += size) ++local_rows;

    COOMatrix local(local_rows, h.cols);

    const char* line = parse_begin;
    while (line < parse_end) {
        const char* nl = line;
        while (nl < parse_end && *nl != '\n') ++nl;

        int r1 = 0, c1 = 0;
        double v = 0.0;

        if (parse_triplet_line(line, nl, h.is_pattern, r1, c1, v)) {
            int gr = r1 - 1; // 1-based -> 0-based
            int gc = c1 - 1;

            if (gr >= 0 && gr < h.rows && gc >= 0 && gc < h.cols) {
                int owner = gr % size;
                if (owner == rank) {
                    int lr = (gr - rank) / size;
                    local.addEntry(lr, gc, v);
                }
                if (h.is_symmetric && gr != gc) {
                    int gr2 = gc;
                    int gc2 = gr;
                    int owner2 = gr2 % size;
                    if (owner2 == rank) {
                        int lr2 = (gr2 - rank) / size;
                        local.addEntry(lr2, gc2, v);
                    }
                }
            }
        }

        line = (nl < parse_end) ? (nl + 1) : parse_end;
    }

    return local;
}

// ------------------------------------------------------------
// MPI-IO + owner-by-grid redistribution for 2D partitioning
// Each rank reads a chunk, parses triplets, and sends each entry to the
// owning (pr,pc) rank --> owner uses uneven block partitioning
//
// COO stored with LOCAL row/col indices inside the owner's block
// ------------------------------------------------------------
COOMatrix SimpleParallelReader::read_2D_block_mpiio_redistribute(const std::string& filename,
                                                                int Pr, int Pc,
                                                                int my_r, int my_c,
                                                                MPI_Comm comm)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (Pr * Pc != size) {
        throw std::runtime_error("2D reader: Pr*Pc must equal #ranks");
    }

    MMHeaderInfo h = read_mmio_header(filename, comm);

    // My block geometry (uneven blocks)
    const int row_start = block_start_uneven(h.rows, Pr, my_r);
    const int col_start = block_start_uneven(h.cols, Pc, my_c);
    const int local_rows = block_size_uneven(h.rows, Pr, my_r);
    const int local_cols = block_size_uneven(h.cols, Pc, my_c);

    // --- MPI-IO chunk read ---
    MPI_File fh;
    if (MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_File_open failed for: " + filename);
    }

    MPI_Offset file_size = 0;
    MPI_File_get_size(fh, &file_size);
    if (h.data_start >= file_size) {
        MPI_File_close(&fh);
        throw std::runtime_error("Invalid data_start >= file_size for: " + filename);
    }

    MPI_Offset data_bytes = file_size - h.data_start;
    MPI_Offset chunk = data_bytes / size;
    MPI_Offset start = h.data_start + (MPI_Offset)rank * chunk;
    MPI_Offset end   = (rank == size - 1) ? file_size
                                          : (h.data_start + (MPI_Offset)(rank + 1) * chunk);

    const MPI_Offset overlap = (MPI_Offset)(1 << 20);
    MPI_Offset read_start = (rank == 0) ? start : std::max(h.data_start, start - overlap);
    MPI_Offset read_end   = (rank == size - 1) ? end   : std::min(file_size, end + overlap);
    MPI_Offset read_nbytes = read_end - read_start;
    if (read_nbytes < 0) read_nbytes = 0;

    std::vector<char> buf((size_t)read_nbytes + 1, '\0');
    MPI_Status st;
    if (MPI_File_read_at_all(fh, read_start, buf.data(), (int)read_nbytes, MPI_BYTE, &st) != MPI_SUCCESS) {
        MPI_File_close(&fh);
        throw std::runtime_error("MPI_File_read_at_all failed for: " + filename);
    }
    MPI_File_close(&fh);

    const char* b = buf.data();
    const char* b_end = b + read_nbytes;

    const char* parse_begin = b + (start - read_start);
    const char* parse_end   = b + (end   - read_start);

    if (rank != 0) {
        const char* p = parse_begin;
        while (p < b_end && *p != '\n') ++p;
        if (p < b_end) ++p;
        parse_begin = p;
    }
    if (rank != size - 1) {
        const char* p = parse_end;
        while (p < b_end && *p != '\n') ++p;
        if (p < b_end) ++p;
        parse_end = p;
    } else {
        parse_end = b_end;
    }

    if (parse_begin < b) parse_begin = b;
    if (parse_end > b_end) parse_end = b_end;
    if (parse_begin > parse_end) parse_begin = parse_end;

    // --- Send buffers per destination rank ---
    std::vector<std::vector<int>> sr(size), sc(size);
    std::vector<std::vector<double>> sv(size);

    auto push_entry = [&](int gr, int gc, double v) {
        if (gr < 0 || gr >= h.rows || gc < 0 || gc >= h.cols) return;

        const int pr = owner_block_uneven(h.rows, Pr, gr);
        const int pc = owner_block_uneven(h.cols, Pc, gc);
        const int dest = pr * Pc + pc;

        const int r0 = block_start_uneven(h.rows, Pr, pr);
        const int c0 = block_start_uneven(h.cols, Pc, pc);

        // store as LOCAL indices for the DEST
        sr[dest].push_back(gr - r0);
        sc[dest].push_back(gc - c0);
        sv[dest].push_back(v);
    };

    const char* line = parse_begin;
    while (line < parse_end) {
        const char* nl = line;
        while (nl < parse_end && *nl != '\n') ++nl;

        int r1 = 0, c1 = 0;
        double v = 0.0;

        if (parse_triplet_line(line, nl, h.is_pattern, r1, c1, v)) {
            int gr = r1 - 1;
            int gc = c1 - 1;
            push_entry(gr, gc, v);
            if (h.is_symmetric && gr != gc) {
                push_entry(gc, gr, v);
            }
        }

        line = (nl < parse_end) ? (nl + 1) : parse_end;
    }

    // --- Alltoallv redistribution ---
    std::vector<int> sendcounts(size, 0), recvcounts(size, 0);
    for (int p = 0; p < size; ++p) sendcounts[p] = (int)sr[p].size();

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::vector<int> sdispls(size, 0), rdispls(size, 0);
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, sdispls.begin() + 1);
    std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, rdispls.begin() + 1);

    const int send_total = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
    const int recv_total = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

    std::vector<int> flat_sr(send_total), flat_sc(send_total);
    std::vector<double> flat_sv(send_total);

    for (int p = 0; p < size; ++p) {
        int off = sdispls[p];
        std::copy(sr[p].begin(), sr[p].end(), flat_sr.begin() + off);
        std::copy(sc[p].begin(), sc[p].end(), flat_sc.begin() + off);
        std::copy(sv[p].begin(), sv[p].end(), flat_sv.begin() + off);
    }

    std::vector<int> rcv_r(recv_total), rcv_c(recv_total);
    std::vector<double> rcv_v(recv_total);

    MPI_Alltoallv(flat_sr.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  rcv_r.data(), recvcounts.data(), rdispls.data(), MPI_INT, comm);

    MPI_Alltoallv(flat_sc.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  rcv_c.data(), recvcounts.data(), rdispls.data(), MPI_INT, comm);

    MPI_Alltoallv(flat_sv.data(), sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                  rcv_v.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, comm);

    // --- Build local COO (LOCAL indices for my block) ---
    COOMatrix local(local_rows, local_cols);
    for (int k = 0; k < recv_total; ++k) {
        int lr = rcv_r[k];
        int lc = rcv_c[k];
        if (lr >= 0 && lr < local_rows && lc >= 0 && lc < local_cols) {
            local.addEntry(lr, lc, rcv_v[k]);
        }
    }

    return local;
}
