#include "mpi_benchmark.hpp"
#include <fstream>
#include <chrono>
#include <mpi.h>

MPITiming MPIBenchmark::benchmark_spmv(const DistributedMatrix& A,
                                       const std::vector<double>& x,
                                       int runs) {
    std::vector<double> y;
    std::vector<double> times;

    for (int r = 0; r < runs; ++r) {
        MPI_Barrier(A.comm);

        auto t0 = std::chrono::high_resolution_clock::now();
        A.spmv(x, y);
        MPI_Barrier(A.comm);
        auto t1 = std::chrono::high_resolution_clock::now();

        double dt =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(dt);
    }

    double local_avg =
        std::accumulate(times.begin(), times.end(), 0.0) / runs;

    double global_avg = 0.0;
    MPI_Reduce(&local_avg, &global_avg, 1,
               MPI_DOUBLE, MPI_MAX, 0, A.comm);

    return {global_avg};
}

// ================= CSV =================
void MPIBenchmark::write_csv_header(const std::string& filename) {
    std::ofstream f(filename);
    f << "matrix,partitioning,mpi_procs,omp_threads,avg_spmv_ms\n";
}

void MPIBenchmark::write_csv_row(const std::string& filename,
                                 const std::string& matrix,
                                 const std::string& partitioning,
                                 int mpi_procs,
                                 int omp_threads,
                                 const MPITiming& t) {
    std::ofstream f(filename, std::ios::app);
    f << matrix << ","
      << partitioning << ","
      << mpi_procs << ","
      << omp_threads << ","
      << t.spmv_time_ms << "\n";
}
