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

void COOMatrix::generateRandomSparse(int n, double sparsity) {
    rows = n;
    cols = n;
    nnz = static_cast<int>(n * n * (1.0 - sparsity));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);
    std::uniform_int_distribution<int> index_dist(0, n-1);
    
    row_idx.resize(nnz);
    col_idx.resize(nnz);
    values.resize(nnz);
    
    for (int i = 0; i < nnz; i++) {
        row_idx[i] = index_dist(gen);
        col_idx[i] = index_dist(gen);
        values[i] = value_dist(gen);
    }
}

void CSRMatrix::convertFromCOO(const COOMatrix& coo) {
    rows = coo.rows;
    cols = coo.cols;
    nnz = coo.nnz;
    
    values.resize(nnz);
    col_idx.resize(nnz);
    row_ptr.resize(rows + 1, 0);
    
    // Count non-zeros per row
    for (int i = 0; i < nnz; i++) {
        row_ptr[coo.row_idx[i] + 1]++;
    }
    
    // Cumulative sum
    for (int i = 0; i < rows; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }
    
    // Fill values and column indices
    std::vector<int> current_pos(row_ptr.begin(), row_ptr.begin() + rows);
    
    for (int i = 0; i < nnz; i++) {
        int row = coo.row_idx[i];
        int pos = current_pos[row];
        values[pos] = coo.values[i];
        col_idx[pos] = coo.col_idx[i];
        current_pos[row]++;
    }
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
