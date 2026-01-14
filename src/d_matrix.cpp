#include "d_matrix.hpp"
#include "p_reader.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std::chrono;

// -----------------------------------------------------------------------------
// Helpers for uneven block partitioning (2D)
// -----------------------------------------------------------------------------
static inline int block_start_uneven(int n, int P, int p) {
    int base = n / P;
    int rem  = n % P;
    return p * base + std::min(p, rem);
}
static inline int block_size_uneven(int n, int P, int p) {
    int base = n / P;
    int rem  = n % P;
    return base + (p < rem ? 1 : 0);
}

// -----------------------------------------------------------------------------
// Helpers for Weak Scheduling Matrix Generation
// -----------------------------------------------------------------------------
static COOMatrix generate_local_weak_1d_cyclic(int global_rows, int global_cols,
                                               int nnz_per_rank,
                                               int rank, int size,
                                               uint64_t seed)
{
    // local rows for cyclic distribution
    int local_rows = 0;
    for (int i = rank; i < global_rows; i += size) ++local_rows;

    COOMatrix local(local_rows, global_cols);

    std::mt19937_64 rng(seed + (uint64_t)rank * 1315423911ULL);
    std::uniform_int_distribution<int> row_dist(0, local_rows - 1);
    std::uniform_int_distribution<int> col_dist(0, global_cols - 1);

    for (int k = 0; k < nnz_per_rank; ++k) {
        int lr = row_dist(rng);
        int gc = col_dist(rng);
        double v = 1.0; // or random value
        local.addEntry(lr, gc, v);
    }

    return local;
}

static COOMatrix generate_local_weak_2d_block(int global_rows, int global_cols,
                                             int nnz_per_rank,
                                             int Pr, int Pc,
                                             int my_r, int my_c,
                                             int rank,
                                             uint64_t seed)
{
    int local_rows = block_size_uneven(global_rows, Pr, my_r);
    int local_cols = block_size_uneven(global_cols, Pc, my_c);

    COOMatrix local(local_rows, local_cols);

    std::mt19937_64 rng(seed + (uint64_t)rank * 11400714819323198485ULL);
    std::uniform_int_distribution<int> rdist(0, local_rows - 1);
    std::uniform_int_distribution<int> cdist(0, local_cols - 1);

    for (int k = 0; k < nnz_per_rank; ++k) {
        int lr = rdist(rng);
        int lc = cdist(rng);
        double v = 1.0;
        local.addEntry(lr, lc, v);
    }

    return local;
}

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------
DistributedMatrix::DistributedMatrix(const COOMatrix& global,
                                     Partitioning part,
                                     MPI_Comm world,
                                     bool already_distributed)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

#ifdef _OPENMP
    omp_num_threads = omp_get_max_threads();
#else
    omp_num_threads = 1;
#endif

    global_rows = global.rows;
    global_cols = global.cols;

    if (part == Partitioning::TwoD && size == 1) {
        part = Partitioning::OneD;
    }

    if (!already_distributed) {
        
        // rank0 holds full COO (caller loads it), distribute entries to owners (baseline req)

        int dims_bcast[3] = {global.rows, global.cols, global.nnz};
        MPI_Bcast(dims_bcast, 3, MPI_INT, 0, world);
        global_rows = dims_bcast[0];
        global_cols = dims_bcast[1];

        // Distribute nnz by cyclic row owner(row)=row%P, then build local COO with local row indices
        std::vector<int> sendcounts(size, 0), displs_(size, 0);
        std::vector<int> flat_r, flat_c;
        std::vector<double> flat_v;

        if (rank == 0) {
            std::vector<std::vector<int>> br(size), bc(size);
            std::vector<std::vector<double>> bv(size);

            for (int k = 0; k < global.nnz; ++k) {
                int gr = global.row_idx[k];
                int owner = gr % size;
                br[owner].push_back(gr);
                bc[owner].push_back(global.col_idx[k]);
                bv[owner].push_back(global.values[k]);
            }

            for (int p = 0; p < size; ++p) sendcounts[p] = (int)br[p].size();
            std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, displs_.begin() + 1);

            int total = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
            flat_r.resize(total);
            flat_c.resize(total);
            flat_v.resize(total);

            for (int p = 0; p < size; ++p) {
                int off = displs_[p];
                std::copy(br[p].begin(), br[p].end(), flat_r.begin() + off);
                std::copy(bc[p].begin(), bc[p].end(), flat_c.begin() + off);
                std::copy(bv[p].begin(), bv[p].end(), flat_v.begin() + off);
            }
        }

        int local_nnz = 0;
        MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, world);

        std::vector<int> r_local(local_nnz), c_local(local_nnz);
        std::vector<double> v_local(local_nnz);

        MPI_Scatterv(rank == 0 ? flat_r.data() : nullptr, sendcounts.data(), displs_.data(), MPI_INT,
                     r_local.data(), local_nnz, MPI_INT, 0, world);
        MPI_Scatterv(rank == 0 ? flat_c.data() : nullptr, sendcounts.data(), displs_.data(), MPI_INT,
                     c_local.data(), local_nnz, MPI_INT, 0, world);
        MPI_Scatterv(rank == 0 ? flat_v.data() : nullptr, sendcounts.data(), displs_.data(), MPI_DOUBLE,
                     v_local.data(), local_nnz, MPI_DOUBLE, 0, world);

        int local_rows_calc = 0;
        for (int i = rank; i < global_rows; i += size) local_rows_calc++;

        COOMatrix local_coo(local_rows_calc, global_cols);
        for (int k = 0; k < local_nnz; ++k) {
            int lr = (r_local[k] - rank) / size;
            local_coo.addEntry(lr, c_local[k], v_local[k]);
        }

        initialize_partitioning(local_coo, part, world, true);
    } else {
        initialize_partitioning(global, part, world, true);
    }
}

DistributedMatrix::DistributedMatrix(const std::string& filename,
                                     Partitioning part,
                                     MPI_Comm world)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

#ifdef _OPENMP
    omp_num_threads = omp_get_max_threads();
#else
    omp_num_threads = 1;
#endif

    COOMatrix local = read_local_portion(filename, part, world);

    // IMPORTANT!!! better to acquire global dims from header, not from local COO dimensions
    // (reader ensures local rows/cols match partitioning)
    initialize_from_local_coo(local, part, world);
}

// Constructor for Weak Scheduling
DistributedMatrix::DistributedMatrix(const COOMatrix& local,
                                     Partitioning part,
                                     MPI_Comm world,
                                     int global_rows_,
                                     int global_cols_)
{
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

#ifdef _OPENMP
    omp_num_threads = omp_get_max_threads();
#else
    omp_num_threads = 1;
#endif

    global_rows = global_rows_;
    global_cols = global_cols_;

    if (part == Partitioning::TwoD && size == 1) part = Partitioning::OneD;

    // local is already distributed
    initialize_partitioning(local, part, world, true);
}

DistributedMatrix DistributedMatrix::FromFileParallel(const std::string& filename,
                                                      Partitioning part,
                                                      MPI_Comm world)
{
    return DistributedMatrix(filename, part, world);
}

// -----------------------------------------------------------------------------
// Reader wrapper
// -----------------------------------------------------------------------------
COOMatrix DistributedMatrix::read_local_portion(const std::string& filename,
                                                Partitioning part,
                                                MPI_Comm world) const
{
    if (part == Partitioning::OneD || size == 1) {
        return SimpleParallelReader::read_1D_cyclic_mpiio(filename, rank, size, world);
    }

    int grid[2] = {0, 0};
    MPI_Dims_create(size, 2, grid);
    int Pr_local = grid[0];
    int Pc_local = grid[1];

    int my_r_local = rank / Pc_local;
    int my_c_local = rank % Pc_local;

    return SimpleParallelReader::read_2D_block_mpiio_redistribute(
        filename, Pr_local, Pc_local, my_r_local, my_c_local, world
    );
}
// -----------------------------------------------------------------------------
// Partition initialization
// -----------------------------------------------------------------------------
void DistributedMatrix::initialize_from_local_coo(const COOMatrix& local_coo,
                                                  Partitioning part,
                                                  MPI_Comm world)
{
    //============================== IMPORTANT ============================
    // set global_rows/global_cols in initialize_partitioning by broadcasting sizes in the 2D setup.
    //=====================================================================
    initialize_partitioning(local_coo, part, world, true);
}

void DistributedMatrix::initialize_partitioning(const COOMatrix& matrix_data,
                                                Partitioning part,
                                                MPI_Comm world,
                                                bool is_already_distributed)
{
    // If we entered here with local COO, global dims may not be set correctly
    // For 1D, matrix_data.cols is global cols, matrix_data.rows is local rows
    // For 2D, both are local. We compute global dims consistently below
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    comm = world;

    // Reset comms
    row_comm = MPI_COMM_NULL;
    col_comm = MPI_COMM_NULL;

    if (part == Partitioning::OneD || size == 1) {
        // 1D cyclic rows: global_cols is the COO columns (global),
        // global_rows inferred from local_rows across ranks.
        local_rows = matrix_data.rows;
        local_cols = matrix_data.cols;

        // Infer global rows by summing local_rows across ranks
        int sum_rows = 0;
        MPI_Allreduce(&local_rows, &sum_rows, 1, MPI_INT, MPI_SUM, comm);
        global_rows = sum_rows;
        global_cols = local_cols;

        local_csr = CSRMatrix(matrix_data);
        return;
    }

    // ---------------- 2D ----------------
    dims[0] = 0; dims[1] = 0;
    MPI_Dims_create(size, 2, dims);
    Pr = dims[0];
    Pc = dims[1];

    int periods[2] = {0, 0};
    MPI_Cart_create(world, 2, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords); 

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    coords[0] = rank / Pc;
    coords[1] = rank % Pc;
    my_r = coords[0];
    my_c = coords[1];

    row_start = block_start_uneven(matrix_data.rows * Pr, Pr, my_r);
    col_start = block_start_uneven(matrix_data.cols * Pc, Pc, my_c);

    // Build row/col communicators
    MPI_Comm_split(comm, my_r, my_c, &row_comm);
    MPI_Comm_split(comm, my_c, my_r, &col_comm);

    // Compute my local block sizes from the local COO itself 
    local_rows = matrix_data.rows;
    local_cols = matrix_data.cols;

    // compute global dims by reducing starts+sizes
    // all ranks in column 0 contribute local_rows to total rows --> take max
    int rows_in_col = 0;
    int cols_in_row = 0;

    int col0 = (my_c == 0) ? local_rows : 0;
    MPI_Allreduce(&col0, &rows_in_col, 1, MPI_INT, MPI_SUM, row_comm); // row_comm sums across cols -> only col0 participates

    int row0 = (my_r == 0) ? local_cols : 0;
    MPI_Allreduce(&row0, &cols_in_row, 1, MPI_INT, MPI_SUM, col_comm);

    // broadcast within comm to set global dims consistently
    global_rows = rows_in_col;
    global_cols = cols_in_row;
    MPI_Bcast(&global_rows, 1, MPI_INT, 0, col_comm); // from my_r==0?
    MPI_Bcast(&global_cols, 1, MPI_INT, 0, row_comm);

    // set starts using the known global dims and uneven partition
    row_start = block_start_uneven(global_rows, Pr, my_r);
    col_start = block_start_uneven(global_cols, Pc, my_c);

    local_csr = CSRMatrix(matrix_data);
}

// -----------------------------------------------------------------------------
// SpMV
// -----------------------------------------------------------------------------
void DistributedMatrix::spmv(const std::vector<double>& x_global,
                             std::vector<double>& y_local,
                             double* comm_time_ms,
                             double* comp_time_ms) const
{
    if (row_comm == MPI_COMM_NULL) {
        // 1D mode
        y_local.resize(local_rows);

        auto comp_start = high_resolution_clock::now();

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int global_col = local_csr.col_idx[j];
                sum += local_csr.values[j] * x_global[global_col];
            }
            y_local[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();
        if (comp_time_ms) *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        if (comm_time_ms) *comm_time_ms = 0.0;
        return;
    }

    // 2D mode
    if (Pr == 1) {
        // effectively 1D column distribution
        y_local.resize(local_rows);
        auto comp_start = high_resolution_clock::now();

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int local_col = local_csr.col_idx[j];
                sum += local_csr.values[j] * x_global[col_start + local_col];
            }
            y_local[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();
        if (comp_time_ms) *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        if (comm_time_ms) *comm_time_ms = 0.0;
        return;
    }

    if (Pc == 1) {
        // effectively 1D row distribution using 2D infrastructure
        y_local.resize(local_rows);

        auto comp_start = high_resolution_clock::now();

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
                int global_col = local_csr.col_idx[j];
                sum += local_csr.values[j] * x_global[global_col];
            }
            y_local[i] = sum;
        }

        auto comp_end = high_resolution_clock::now();
        if (comp_time_ms) *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
        if (comm_time_ms) *comm_time_ms = 0.0;
        return;
    }

    // ============================================================
    // TRUE 2D PARTITIONING (Pr > 1 && Pc > 1)
    // ============================================================

    // PHASE 1: Broadcast X blocks along columns
    std::vector<double> x_block(local_cols, 0.0);

    auto comm1_start = high_resolution_clock::now();

    // Process at row 0 in each column extracts its x block
    if (my_r == 0) {
        if (col_start < (int)x_global.size()) {
            int copy_size = std::min(local_cols, (int)x_global.size() - col_start);
            std::copy(x_global.begin() + col_start,
                      x_global.begin() + col_start + copy_size,
                      x_block.begin());
        }
    }

    MPI_Bcast(x_block.data(), local_cols, MPI_DOUBLE, 0, col_comm);

    auto comm1_end = high_resolution_clock::now();

    // PHASE 2: Local computation
    std::vector<double> y_partial(local_rows, 0.0);

    auto comp_start = high_resolution_clock::now();

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        for (int j = local_csr.row_ptr[i]; j < local_csr.row_ptr[i + 1]; ++j) {
            int local_col = local_csr.col_idx[j]; // local
            sum += local_csr.values[j] * x_block[local_col];
        }
        y_partial[i] = sum;
    }

    auto comp_end = high_resolution_clock::now();

    // PHASE 3: Reduce results along rows (to column-root only)
    auto comm2_start = high_resolution_clock::now();

    y_local.resize(local_rows, 0.0);

    // Only my_c==0 receives the reduced y for this row-block!!!
    MPI_Reduce(y_partial.data(), y_local.data(), local_rows,
               MPI_DOUBLE, MPI_SUM, 0, row_comm);

    // Keep others zeroed (safety)
    if (my_c != 0) {
        std::fill(y_local.begin(), y_local.end(), 0.0);
    }

    auto comm2_end = high_resolution_clock::now();

    if (comm_time_ms) {
        *comm_time_ms =
            duration<double, std::milli>(comm1_end - comm1_start).count() +
            duration<double, std::milli>(comm2_end - comm2_start).count();
    }
    if (comp_time_ms) {
        *comp_time_ms = duration<double, std::milli>(comp_end - comp_start).count();
    }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
size_t DistributedMatrix::getLocalMemoryUsage() const {
    size_t mem = 0;
    mem += local_csr.values.capacity() * sizeof(double);
    mem += local_csr.col_idx.capacity() * sizeof(int);
    mem += local_csr.row_ptr.capacity() * sizeof(int);
    return mem;
}

void DistributedMatrix::printInfo() const {
    std::cout << "Rank " << rank << ": "
              << "Global " << global_rows << "x" << global_cols
              << ", Local " << local_rows << "x" << local_cols
              << ", CSR nnz=" << local_csr.nnz;
    if (row_comm != MPI_COMM_NULL) {
        std::cout << " (2D " << Pr << "x" << Pc << ")";
    } else {
        std::cout << " (1D)";
    }
    std::cout << std::endl;
}

double DistributedMatrix::get_load_imbalance() const {
    size_t local_nnz = local_csr.nnz;
    size_t max_nnz, min_nnz, total_nnz;

    MPI_Allreduce(&local_nnz, &max_nnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&local_nnz, &min_nnz, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_nnz, &total_nnz, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    double avg_nnz = total_nnz / (double)size;
    if (avg_nnz == 0) return 0.0;
    return (max_nnz - min_nnz) / avg_nnz;
}

size_t DistributedMatrix::get_local_nnz() const {
    return local_csr.nnz;
}

void DistributedMatrix::print_partitioning_info() const {
    if (row_comm == MPI_COMM_NULL) {
        if (rank == 0) std::cout << "Partitioning: 1D cyclic rows\n";
        return;
    }
    if (rank == 0) {
        std::cout << "Partitioning: 2D " << Pr << "x" << Pc << " grid\n";
    }
}
