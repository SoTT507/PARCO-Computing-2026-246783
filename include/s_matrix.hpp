#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP
#include "pch.h"

class COOMatrix {
public:
    int rows, cols, nnz;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> values;
    
    COOMatrix() = default;
    COOMatrix(int r, int c) : rows(r), cols(c), nnz(0) {}
    COOMatrix(const std::string& filename) { readMatrixMarket(filename); }
    
    void readMatrixMarket(const std::string& filename,bool keep_1_based = false);
    void addEntry(int i, int j, double val);
    void generateRandomSparse(int n, double sparsity);
    void generateRandomSparseNNZ(int n, double density, int target_nnz = -1);
};

class CSRMatrix {
public:
    int rows, cols, nnz;
    std::vector<double> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    
    CSRMatrix() : rows(0), cols(0), nnz(0) {}
    CSRMatrix(const COOMatrix& coo) { convertFromCOO(coo); }
    
    void convertFromCOO(const COOMatrix& coo);
    void spmv(const std::vector<double>& x, std::vector<double>& y) const;
};

#endif
