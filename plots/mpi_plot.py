import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mpi_spmv_results.csv")

for m in df['matrix'].unique():
    sub = df[df['matrix'] == m]
    plt.figure()
    for p in ['1D', '2D']:
        s = sub[sub['partitioning'] == p]
        plt.plot(s['mpi_procs'], s['avg_spmv_ms'], label=p, marker='o')

    plt.title(m)
    plt.xlabel("MPI processes")
    plt.ylabel("SpMV time [ms]")
    plt.legend()
    plt.grid(True)
    plt.show()
