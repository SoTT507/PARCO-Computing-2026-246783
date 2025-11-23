# Sparse Matrix-Vector Multiplication Benchmark



## Project Overview
#### Assignment: 
- **1.** Develop a code which reads a matrix in the Matrix Format and convert to CSR
- **2.** Multiply the Matrix with a randomly generated array(Sequential code)
- **3.** Design a parallel code on a shared-memory system (OpenMP has been used)

#### Benchmarking: 
- Report of multiplication results [ms] across multiple matrices with different degrees of sparsity
- Comparison between Parallel and Sequential code by increasing the number of threads
- Evaluation of different scheduling strategies
- Bottleneck identification 

#### Prerequisites
- C++11 compatible compiler (or later)
- CMake 3.10+
- OpenMP support
- Python 3.8+ (for data plotting)

#### Download
```bash
git clone https://github.com/SoTT507/PARCO-Computing-2026-246783
cd PARCO-Computing-2026-246783
```
#### Building
```bash
./build.sh
```
#### Running
```bash
#run the deliverable and store csv data
./d1_SpMV
cd plots
#print the graphs
python3 plot.py

```

#### Build and Run
