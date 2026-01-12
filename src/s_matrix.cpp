#include "s_matrix.hpp"
#include "mmio.h"
void COOMatrix::readMatrixMarket(const std::string& filename, bool keep_1_based) {
  FILE* f = fopen(filename.c_str(), "r");
    if (!f)
        throw std::runtime_error("Cannot open file: " + filename);

    MM_typecode matcode;

    if (mm_read_banner(f, &matcode) != 0)
        throw std::runtime_error("Could not process MatrixMarket banner in: " + filename);

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
        throw std::runtime_error("Only sparse MatrixMarket matrices are supported: " + filename);

    // Size
    if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0)
        throw std::runtime_error("Could not read matrix dimensions: " + filename);

    // Allocate
    row_idx.resize(nnz);
    col_idx.resize(nnz);
    values.resize(nnz);

    bool is_pattern  = mm_is_pattern(matcode);
    bool is_real     = mm_is_real(matcode) || mm_is_integer(matcode);
    bool is_symmetric = mm_is_symmetric(matcode);

    int r, c;
    double v;

    int k = 0;

    // --- Read COO triplets ---
    for (int i = 0; i < nnz; i++)
    {
        if (is_pattern) {
            // FORMAT: i j
            if (fscanf(f, "%d %d", &r, &c) != 2)
                throw std::runtime_error("Invalid pattern entry in: " + filename);
            v = 1.0;
        } else {
            // FORMAT: i j val
            if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3)
                throw std::runtime_error("Invalid numeric entry in: " + filename);
        }

        if (!keep_1_based) { r--; c--; }

        row_idx[k] = r;
        col_idx[k] = c;
        values[k]  = v;
        k++;

        // For symmetric matrices --> duplicate lower triangle automatically
        if (is_symmetric && r != c)
        {
            int rr = r, cc = c;
            if (!keep_1_based) std::swap(rr, cc);
            else std::swap(r, c);

            row_idx.push_back(c);
            col_idx.push_back(r);
            values.push_back(v);
        }
    }

    fclose(f);

    // If symmetric expanded, update nnz
    nnz = row_idx.size();
}

void COOMatrix::addEntry(int i, int j, double val) {
    row_idx.push_back(i);
    col_idx.push_back(j);
    values.push_back(val);
    nnz++;
}

void COOMatrix::generateRandomSparse(int n, double density, int target_nnz = -1) {
    rows = n;
    cols = n;
    
    // Use target_nnz if provided, otherwise calculate from density
    if (target_nnz > 0) {
        nnz = target_nnz;
    } else {
        nnz = static_cast<int>(n * n * density);
    }
    
    if (nnz <= 0) nnz = 1;
    if (nnz > n * n) nnz = n * n;  // Can't have more nnz than total elements

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);
    std::uniform_int_distribution<int> row_dist(0, n-1);
    std::uniform_int_distribution<int> col_dist(0, n-1);

    row_idx.resize(nnz);
    col_idx.resize(nnz);
    values.resize(nnz);

    // Use a set to avoid duplicate entries
    std::unordered_set<int64_t> used_positions;
    
    for (int i = 0; i < nnz; i++) {
        int attempts = 0;
        int64_t pos;
        
        // Ensure unique positions (optional but good for realistic matrices)
        do {
            int r = row_dist(gen);
            int c = col_dist(gen);
            pos = static_cast<int64_t>(r) * n + c;
            
            if (++attempts > 100) {
                // If too many attempts, allow duplicates
                break;
            }
        } while (used_positions.find(pos) != used_positions.end());
        
        used_positions.insert(pos);
        
        // Extract row and column from position
        row_idx[i] = pos / n;
        col_idx[i] = pos % n;
        values[i] = value_dist(gen);
    }
}
// s_matrix_debug.cpp - Update convertFromCOO function
void CSRMatrix::convertFromCOO(const COOMatrix& coo) {
    std::cout << "CSR::convertFromCOO called: coo.rows=" << coo.rows
              << ", coo.cols=" << coo.cols << ", coo.nnz=" << coo.nnz << std::endl;

    rows = coo.rows;
    cols = coo.cols;
    nnz = coo.nnz;

    // SAFETY CHECK: Handle empty matrix
    if (rows <= 0) {
        std::cout << "CSR::convertFromCOO: rows <= 0, creating empty matrix" << std::endl;
        values.clear();
        col_idx.clear();
        row_ptr.resize(1, 0);
        return;
    }

    if (nnz <= 0) {
        std::cout << "CSR::convertFromCOO: nnz <= 0, creating empty matrix" << std::endl;
        values.resize(0);
        col_idx.resize(0);
        row_ptr.resize(rows + 1, 0);
        return;
    }

    // Validate input indices
    std::cout << "CSR::convertFromCOO: Validating indices..." << std::endl;
    for (int i = 0; i < nnz; i++) {
        if (coo.row_idx[i] < 0 || coo.row_idx[i] >= rows) {
            std::cerr << "ERROR in convertFromCOO: Invalid row index "
                      << coo.row_idx[i] << " at position " << i
                      << " (max: " << rows-1 << ")" << std::endl;
            throw std::runtime_error("Invalid row index in COO matrix");
        }
        if (coo.col_idx[i] < 0 || coo.col_idx[i] >= cols) {
            std::cerr << "ERROR in convertFromCOO: Invalid column index "
                      << coo.col_idx[i] << " at position " << i
                      << " (max: " << cols-1 << ")" << std::endl;
            throw std::runtime_error("Invalid column index in COO matrix");
        }
    }

    std::cout << "CSR::convertFromCOO: Allocating memory..." << std::endl;
    values.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(rows + 1, 0);

    std::cout << "CSR::convertFromCOO: Counting non-zeros per row..." << std::endl;
    // Count non-zeros per row (with bounds checking)
    for (int i = 0; i < nnz; i++) {
        int row = coo.row_idx[i];
        // Double-check bounds
        if (row >= 0 && row < rows) {
            row_ptr[row + 1]++;
        } else {
            std::cerr << "ERROR: Row index " << row << " out of bounds!" << std::endl;
            throw std::runtime_error("Row index out of bounds");
        }
    }

    std::cout << "CSR::convertFromCOO: Cumulative sum..." << std::endl;
    // Cumulative sum
    for (int i = 0; i < rows; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Fill values and column indices
    std::cout << "CSR::convertFromCOO: Filling values..." << std::endl;
    std::vector<int> current_pos(row_ptr.begin(), row_ptr.begin() + rows);

    for (int i = 0; i < nnz; i++) {
        int row = coo.row_idx[i];
        int pos = current_pos[row];

        if (pos >= 0 && pos < nnz) {
            values[pos] = coo.values[i];
            col_idx[pos] = coo.col_idx[i];
            current_pos[row]++;
        } else {
            std::cerr << "ERROR: Invalid position " << pos << " for row " << row << std::endl;
            throw std::runtime_error("Invalid position in CSR conversion");
        }
    }

    // Final validation
    if (row_ptr[rows] != nnz) {
        std::cerr << "ERROR: CSR conversion failed - nnz mismatch: "
                  << row_ptr[rows] << " != " << nnz << std::endl;
        throw std::runtime_error("CSR conversion nnz mismatch");
    }

    std::cout << "CSR::convertFromCOO: Success! Created "
              << rows << "x" << cols << " CSR with " << nnz << " non-zeros" << std::endl;
}

void CSRMatrix::spmv(const std::vector<double>& x, std::vector<double>& y) const {
    if (x.size() != static_cast<size_t>(cols)) {
        throw std::invalid_argument("Vector size doesn't match matrix columns");
    }

    y.resize(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_idx[j]];
        }
    }
}
