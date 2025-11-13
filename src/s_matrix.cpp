#include "s_matrix.hpp"

void COOMatrix::readMatrixMarket(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }
    
    // Read matrix dimensions
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;
    row_idx.resize(nnz);
    col_idx.resize(nnz);
    values.resize(nnz);
    
    // Read non-zero entries
    for (int i = 0; i < nnz; i++) {
        int row, col;
        double value;
        file >> row >> col >> value;
        row_idx[i] = row - 1;  // Convert to 0-based
        col_idx[i] = col - 1;
        values[i] = value;
    }
    
    file.close();
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
