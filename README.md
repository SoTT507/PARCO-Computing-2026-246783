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
## For Linux (Ubuntu - Debian-based)
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
```bash
./build_and_run.sh
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

#move the executable in the main folder along with mmio static library
mv ./d1_SpMV ../
mv ./libmmio.a ../

cd ..
./d1_SpMV

#To print the graphs
cd plots
python3 plot.py

```
## Notes on implementation which uses T.P. software
The methods to read the matrices in the Matrix Market format make use of the library routines available at https://math.nist.gov/MatrixMarket/mmio-c.html
found under **Source Code** section

#### To run on the cluster
First of all, load the modules to meet the requirements:
```bash
module load cmake-3.15.4
module load python-3.8.13
```
```bash 
#In the root folder
./runBM.pbs

#to plot the graphs
cd plots
python3 plot.py
```
