# Distributed Sparse Matrix-Vector Multiplication (SpMV) with MPI

## Project Overview
- Implementation of a distributed-memory SpMV using MPI and focusing on data distribution, communication, and scalability

### Features:
#### Matrix Reading:
- Rank 0 reads the entire Matrix Market file and distributes matrix entries to all processes
- To parse and read the mmio library has been used (https://math.nist.gov/MatrixMarket/mmio-c.html)
#### Data Partitioning 
- 1D module (cyclic) Partitioning
- Row Ownership rule: owner(i)=imodP where P is the number of Processes and i is the row index
- Each process stores all nonzeros belonging to its rows
- Local + global index representations
- 2D partitioning (process grid)

1D and 2D block layouts for 6 processes. Each color represents a process.
![alt text](https://github.com/SoTT507/PARCO-Computing-2026-246783/blob/main/partitioning.png?raw=true)

#### Data Structure
- Each process should build a local sparse representation, e.g: COO --> CSR conversion
- Local row index: Compute the local index: e.g. global index (index of the entire matrix) / number of processes (integer part)
- Local SpMV uses Multi-threading (MPI+X)
**Reuse of SpMV computation methods developed in Deliverable 1**

#### 1D SpMV Communication Pattern
- each rank computes y_i for its owned rows
- Communication required to fetch remote x_j entries if any
- Identify required remote column indices ("ghost entries")
- Exchange vector values with other ranks
- Perform local SpMV
- (Optional) gather or keep distributed result

#### Performance Evaluation
##### Strong Scaling
- Fixed matrix size
- Increase number of processes up to 128
- Use real matrices

##### Weak Scaling
- Increase matrix size with number of processes
- Use synthetic matrices (Random)

#### Metrics
- Execution time per SpMV
- Speedup and efficiency (per rank)
- FLOPs
- Communication vs computation breakdown
- Memory footprint per rank
