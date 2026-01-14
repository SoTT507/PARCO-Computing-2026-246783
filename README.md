# Distributed Sparse Matrix–Vector Multiplication (SpMV) with MPI

## Project Overview
This project implements a distributed-memory Sparse Matrix–Vector Multiplication (SpMV)
using MPI, with a focus on data distribution, communication patterns, and scalability.
Both 1D and 2D partitioning strategies are supported, together with MPI+OpenMP hybrid
parallelism.

The project is developed as part of the PARCO (Parallel Computing) course and extends
the local SpMV implementation developed in Deliverable 1.

---

## Features
- Distributed-memory SpMV with MPI 
- 1D cyclic row partitioning 
- Local sparse storage (CSR)
- Reuse of Deliverable 1 SpMV
- Strong scaling on fixed matrices
- Weak scaling on synthetic matrices
- MPI+OpenMP hybrid model
- MPI-IO (`MPI_File_read_at_all`)
- Header + line-boundary handling
- 2D partitioning
- Owner-by-grid redistribution
- True 2D SpMV algorithm
- Communication/computation breakdown
- Memory usage per rank

### Partially Implemnted / Simplified
- Ghost entries in 1D SpMV: Vector assumed replicated; no explicit halo exchange
- Final result gathering: 2D result only valid on `my_c == 0`
- MPI topology optimization: Cartesian topology used, but no hardware-aware mapping


### Data Partitioning

<<<<<<< HEAD
#### 1D Partitioning (Required)
- Row-wise cyclic distribution
  where \( P \) is the number of MPI processes and \( i \) is the global row index.
- Each process stores all nonzeros belonging to its owned rows.
- Explicit global-to-local row index mapping is performed.

#### 2D Partitioning (Bonus)
- A 2D process grid is created using `MPI_Dims_create`.
- Uneven block partitioning is supported when matrix dimensions are not divisible by the
  process grid dimensions.
- Owner-by-grid redistribution is implemented: nonzeros are assigned to the owning
  (row-block, column-block) rank.
- Local matrices store **local row and local column indices**.

Illustration of 1D and 2D layouts (example with 6 processes):

![Partitioning schemes](https://github.com/SoTT507/PARCO-Computing-2026-246783/blob/main/partitioning.png?raw=true)

---

### Metrics Collected
- Average SpMV execution time.
- Communication time vs computation time.
- Speedup and efficiency (strong scaling).
- Weak scaling efficiency.
- Floating-point performance (GFLOPs).
- Memory footprint per MPI rank (in prints).


# Sparse Matrix-Vector Multiplication Benchmark

## Usage

### The program has 2 available flags:
```bash
# 1). (recommended, much shorter execution time and better performance)
--parallel-io
# Uses filename constructor for MPI-IO reader implementation

#2).
--compare
# rank0 reads entire matrix, then the constructor distributes nnz
```

## Build & Run

### Prerequisites
- C++17 compatible compiler (tested on GCC 12.2.0)
- CMake ≥ 3.24 (tested on CMake 3.24.3)
- MPI implementation (tested on OpenMPI 4.1.4)
- OpenMP support
- Python ≥ 3.8 (for plotting)


#### Download
```bash
git clone https://github.com/SoTT507/PARCO-Computing-2026-246783
cd PARCO-Computing-2026-246783
```
## For Linux (Ubuntu - Debian-based)
#### Building
```bash
./build.sh
```
#### Running
```bash
#run the deliverable and store csv data
mpiexec -n 4 ./d2_SpMV --parallel-io
cd plots
#print the graphs
python3 mpi_plot.py

```
#### Build and Run
```bash
./test.sh
```
### IMPORTANT
The code has been developed and tested on Linux systems, therefore running with the commands above on Windows will result in error:
Try the following commands but running is not granted. Linux is the way :)

## Windows & Linux (Ubuntu - Debian-based)

#### Manual Building/Running
```bash
cd build/
cmake --fresh ..
cmake --build .

#move the executable in the main folder
mv ./d2_SpMV ../

cd ..
mpiexec -n 2 ./d2_SpMV --parallel-io

#To print the graphs
cd plots
python3 mpi_plot.py

```
## Notes on implementation which uses T.P. software
The methods to read the matrices in the Matrix Market format make use of the library routines available at https://math.nist.gov/MatrixMarket/mmio-c.html
found under **Source Code** section

## To run on the cluster
First of all, load the modules to meet the requirements and be able to build:
```bash
module load GCCcore/12.2.0
module load zlib/1.2.12-GCCcore-12.2.0
module load binutils/2.39-GCCcore-12.2.0
module load GCC/12.2.0
module load ncurses/6.3-GCCcore-12.2.0
module load bzip2/1.0.8-GCCcore-12.2.0
module load OpenSSL/1.1
module load cURL/7.86.0-GCCcore-12.2.0
module load XZ/5.2.7-GCCcore-12.2.0
module load libarchive/3.6.1-GCCcore-12.2.0
module load CMake/3.24.3-GCCcore-12.2.0
module load numactl/2.0.16-GCCcore-12.2.0
module load libxml2/2.10.3-GCCcore-12.2.0
module load libpciaccess/0.17-GCCcore-12.2.0
module load hwloc/2.8.0-GCCcore-12.2.0
module load libevent/2.1.12-GCCcore-12.2.0
module load UCX/1.13.1-GCCcore-12.2.0
module load libfabric/1.16.1-GCCcore-12.2.0
module load PMIx/4.2.2-GCCcore-12.2.0
module load UCC/1.1.0-GCCcore-12.2.0
module load OpenMPI/4.1.4-GCC-12.2.0
```
#### Build
```bash
#!!!IMPORTANT!!! do not execute test.sh or testHPC.sh (cluster rules) UNLESS you are in an interctive session
./buildHPC.sh
```
#### Submit
```bash 
#In the root folder
qsub runBM.pbs

#to plot the graphs
cd plots
python3 mpi_plot.py
```

#### For Interactive session:
```bash
#Modify to better fit node specs
testHPC.sh
#Or to run basing on physical cores
test.sh

```

## How to
### To modify the matrices selection there are two ways:
- **1** Simply Download and save the desired mtx files inside thirdparty/ folder and modify the main.cpp to benchmark those matrices.
- **2** Change the getmatrices.sh script inside the thirdparty folder and modify the main.cpp to benchmark those matrices.
### To modify the testing logic (how many MPI and OMP per run):
#### qsub
- Access the runBM.pbs
- Modify PBS queue specs if wanted
- Modify CORES_PER_NODE value as preferred (It should respect the number of physical cores available on the machine)
- Modify NODES value as preferred (Is should respect the number of nodes specified by the PBS submission specs above the module loads)
#### interactive && local
- Access the testHPC.sh or test.sh
- Same as for qsub (the test.sh is designed to work in local so core numbers are requested with nproc and of course there are no nodes (I suppose))

### Notes:
- audikw_1 may produce better results than the other matrices

###### After that, rebuild and enjoy
=======
#### Metrics
- Execution time per SpMV
- Speedup and efficiency (per rank)
- FLOPs
- Communication vs computation breakdown
- Memory footprint per rank
