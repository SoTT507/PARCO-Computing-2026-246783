#!/bin/bash
TOTAL_CORES=$(cat $PBS_NODEFILE | wc -l)
NODES=$(uniq $PBS_NODEFILE | wc -l)

MPI_PER_NODE_LIST="1 2 4 8"

for MPI_PER_NODE in $MPI_PER_NODE_LIST
do
    MPI_PROCS=$((MPI_PER_NODE * NODES))
    OMP_THREADS=$((TOTAL_CORES / MPI_PROCS))

    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    echo "MPI=$MPI_PROCS  OMP=$OMP_THREADS"

    mpiexec \
      -n $MPI_PROCS \
      --map-by ppr:${MPI_PER_NODE}:node \
      --bind-to core \
      ./d2_SpMV
done
