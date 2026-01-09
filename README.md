# Distributed Sparse Matrix-Vector Multiplication (SpMV) with MPI

## Project Overview
#### Assignment:
- **1.** Implement a distributed-memory SpMV using MPI
- **2.** Focus on data distribution, communication, and scalability
- **3.** Optional advanced features for bonus points
- - Parallel reading: each rank reads its own file chunk
- - MPI-IO implementation
- - Must handle header and line-boundary alignment correctly

#### Learnings:
- **1.** Read and distribute sparse matrices in Matrix Market format
- **2.** Design 1D or 2D data partitioning strategies
- **3.** Implement distributed-memory SpMV algorithms
- **4.** Analyze strong and weak scalability

### Specs:
- Rank 0 reads the entire Matrix Market file and distributes matrix entries to all processes
- Parallel reading: each rank reads its own file chunk
- MPI-IO implementation (MPI_File_read_at_all)
- Handle header and line-boundary alignment

#### Data Partitioning:
- 1D and 2D Partitioning
- Row ownership rule: owner(i)=imodP where P is the number of Processes and i is the row index
- Each process stores all nonzeros belonging to its rows
- Local + global index representations



