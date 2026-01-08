import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_all_mpi_results(pattern="result_MPI*.csv"):
    """Load all MPI benchmark results CSV files."""
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found with pattern: {pattern}")
        print("Looking for files like: result_MPI2_OMP4.csv")
        # Try alternative pattern
        csv_files = glob.glob("*MPI*.csv")
    
    if not csv_files:
        print("Error: No MPI CSV files found!")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    all_data = []
    for file in csv_files:
        try:
            # Extract MPI and OMP info from filename
            filename = os.path.basename(file)
            # Parse filename like "result_MPI4_OMP8.csv"
            if 'MPI' in filename and 'OMP' in filename:
                parts = filename.replace('result_', '').replace('.csv', '').split('_')
                mpi_procs = int(parts[0].replace('MPI', ''))
                omp_threads = int(parts[1].replace('OMP', ''))
            else:
                # Try to extract from data
                df_temp = pd.read_csv(file, nrows=1)
                if 'mpi_procs' in df_temp.columns and 'omp_threads' in df_temp.columns:
                    mpi_procs = df_temp['mpi_procs'].iloc[0]
                    omp_threads = df_temp['omp_threads'].iloc[0]
                else:
                    print(f"Warning: Could not parse MPI/OMP info from {filename}")
                    continue
            
            df = pd.read_csv(file)
            # Add filename info if not in data
            if 'mpi_procs' not in df.columns:
                df['mpi_procs'] = mpi_procs
            if 'omp_threads' not in df.columns:
                df['omp_threads'] = omp_threads
            
            all_data.append(df)
            print(f"  Loaded {filename}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY:")
    print("="*60)
    print(f"Unique MPI processes: {sorted(combined_df['mpi_procs'].unique())}")
    print(f"Unique OpenMP threads: {sorted(combined_df['omp_threads'].unique())}")
    print(f"Unique matrices: {combined_df['matrix'].unique().tolist()}")
    print(f"Unique partitioning: {combined_df['partitioning'].unique().tolist()}")
    
    return combined_df

def plot_gflops_vs_mpi(df):
    """Plot GFLOP/s vs MPI Processes."""
    
    plt.figure(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))
    
    for i, matrix in enumerate(matrices):
        for partitioning in ['1D', '2D']:
            mask = (df['matrix'] == matrix) & (df['partitioning'] == partitioning)
            data = df[mask].sort_values('mpi_procs')
            
            if not data.empty:
                # Group by MPI processes (average if multiple runs)
                grouped = data.groupby('mpi_procs')['gflops'].mean().reset_index()
                
                label = f"{matrix} ({partitioning})"
                linestyle = '-' if partitioning == '1D' else '--'
                plt.plot(grouped['mpi_procs'], grouped['gflops'], 'o-',
                        color=colors[i], linestyle=linestyle,
                        label=label, markersize=8, linewidth=2)
    
    plt.xlabel('Number of MPI Processes', fontsize=12)
    plt.ylabel('GFLOP/s', fontsize=12)
    plt.title('Performance: GFLOP/s vs MPI Processes', fontsize=14, fontweight='bold')
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gflops_vs_mpi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print best performance for each matrix
    print("\n" + "="*60)
    print("BEST GFLOP/s PERFORMANCE:")
    print("="*60)
    
    for matrix in matrices:
        print(f"\n{matrix}:")
        for partitioning in ['1D', '2D']:
            mask = (df['matrix'] == matrix) & (df['partitioning'] == partitioning)
            if mask.any():
                best_idx = df[mask]['gflops'].idxmax()
                best = df.loc[best_idx]
                print(f"  {partitioning}: {best['gflops']:.2f} GFLOP/s "
                      f"(MPI={best['mpi_procs']}, OMP={best['omp_threads']})")

def plot_hybrid_scaling(df):
    """Plot MPI+OpenMP Hybrid Scaling: Time vs MPI ranks for different thread counts."""
    
    omp_threads = sorted(df['omp_threads'].unique())
   
    #if it's not MPI+X
    if len(omp_threads) == 1:
        print(f"\nOnly one OMP thread count ({omp_threads[0]}) available.")
        print("For hybrid scaling, run with different OMP_NUM_THREADS values.")
        plot_basic_mpi_scaling(df)
        return
    
    print(f"\nAnalyzing hybrid scaling with OMP threads: {omp_threads}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1D partitioning
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(omp_threads)))
    
    for i, omp in enumerate(omp_threads):
        mask = (df['partitioning'] == '1D') & (df['omp_threads'] == omp)
        data = df[mask]
        
        if not data.empty:
            # group by matrix and MPI procs
            for matrix in df['matrix'].unique(): # [:3] to show first 3 matrices
                matrix_data = data[data['matrix'] == matrix].sort_values('mpi_procs')
                if not matrix_data.empty:
                    grouped = matrix_data.groupby('mpi_procs')['avg_ms'].mean().reset_index()
                    
                    if i == 0:
                        label = f"{matrix} ({omp} threads)"
                    else:
                        label = f"{omp} threads"
                    
                    ax1.plot(grouped['mpi_procs'], grouped['avg_ms'], 'o-',
                            color=colors[i], label=label, markersize=6, linewidth=2)
    
    ax1.set_xlabel('Number of MPI Processes')
    ax1.set_ylabel('Time per SpMV (ms)')
    ax1.set_title('1D Partitioning: MPI+OpenMP Hybrid Scaling')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2D partitioning
    ax2 = axes[1]
    
    for i, omp in enumerate(omp_threads):
        mask = (df['partitioning'] == '2D') & (df['omp_threads'] == omp)
        data = df[mask]
        
        if not data.empty:
            for matrix in df['matrix'].unique()[:3]:  # Show first 3 matrices
                matrix_data = data[data['matrix'] == matrix].sort_values('mpi_procs')
                if not matrix_data.empty:
                    grouped = matrix_data.groupby('mpi_procs')['avg_ms'].mean().reset_index()
                    
                    if i == 0:
                        label = f"{matrix} ({omp} threads)"
                    else:
                        label = f"{omp} threads"
                    
                    ax2.plot(grouped['mpi_procs'], grouped['avg_ms'], 's--',
                            color=colors[i], label=label, markersize=6, linewidth=2)
    
    ax2.set_xlabel('Number of MPI Processes')
    ax2.set_ylabel('Time per SpMV (ms)')
    ax2.set_title('2D Partitioning: MPI+OpenMP Hybrid Scaling')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("HYBRID SCALING ANALYSIS:")
    print("="*60)
    
    # compare MPI-only vs MPI+OpenMP for a sample matrix
    sample_matrix = df['matrix'].unique()[0]
    print(f"\nSample matrix: {sample_matrix}")
    
    for partitioning in ['1D', '2D']:
        print(f"\n{partitioning} Partitioning:")
        print("-" * 40)
        
        # MPI-only (1 thread)
        mpi_only = df[(df['matrix'] == sample_matrix) & 
                     (df['partitioning'] == partitioning) &
                     (df['omp_threads'] == 1)].sort_values('mpi_procs')
        
        # MPI+OpenMP (highest thread count)
        max_threads = max(omp_threads)
        mpi_omp = df[(df['matrix'] == sample_matrix) &
                    (df['partitioning'] == partitioning) &
                    (df['omp_threads'] == max_threads)].sort_values('mpi_procs')
        
        if not mpi_only.empty and not mpi_omp.empty:
            for procs in sorted(set(mpi_only['mpi_procs']).intersection(set(mpi_omp['mpi_procs']))):
                time_mpi_only = mpi_only[mpi_only['mpi_procs'] == procs]['avg_ms'].mean()
                time_mpi_omp = mpi_omp[mpi_omp['mpi_procs'] == procs]['avg_ms'].mean()
                
                if time_mpi_omp > 0:
                    speedup = time_mpi_only / time_mpi_omp
                    print(f"  MPI={procs}: MPI-only={time_mpi_only:.1f}ms, "
                          f"MPI+OMP={time_mpi_omp:.1f}ms, Speedup={speedup:.2f}x")

def plot_basic_mpi_scaling(df):
    """Basic MPI scaling plot when only one OMP thread count is available."""
    
    plt.figure(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))
    
    for i, matrix in enumerate(matrices):
        for partitioning in ['1D', '2D']:
            mask = (df['matrix'] == matrix) & (df['partitioning'] == partitioning)
            data = df[mask].sort_values('mpi_procs')
            
            if not data.empty:
                grouped = data.groupby('mpi_procs')['avg_ms'].mean().reset_index()
                label = f"{matrix} ({partitioning})"
                linestyle = '-' if partitioning == '1D' else '--'
                
                plt.plot(grouped['mpi_procs'], grouped['avg_ms'], 'o-',
                        color=colors[i], linestyle=linestyle,
                        label=label, markersize=8, linewidth=2)
    
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Time per SpMV (ms)')
    plt.title(f'MPI Scaling (OMP threads={df["omp_threads"].iloc[0]})')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('mpi_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_speedup_vs_mpi(df):
    """Plot Speedup vs MPI ranks."""
    
    plt.figure(figsize=(12, 8))
    
    matrices = df['matrix'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))
    
    for i, matrix in enumerate(matrices):
        for partitioning in ['1D', '2D']:
            mask = (df['matrix'] == matrix) & (df['partitioning'] == partitioning)
            data = df[mask].sort_values('mpi_procs')
            
            if not data.empty:
                # Find 1 MPI process as baseline
                baseline_data = data[data['mpi_procs'] == 1]
                if baseline_data.empty:
                    # Use minimum MPI count as baseline
                    min_procs = data['mpi_procs'].min()
                    baseline_data = data[data['mpi_procs'] == min_procs]
                
                if not baseline_data.empty:
                    baseline_time = baseline_data['avg_ms'].mean()
                    grouped = data.groupby('mpi_procs')['avg_ms'].mean().reset_index()
                    speedup = baseline_time / grouped['avg_ms']
                    
                    label = f"{matrix} ({partitioning})"
                    linestyle = '-' if partitioning == '1D' else '--'
                    
                    plt.plot(grouped['mpi_procs'], speedup, 's-',
                            color=colors[i], linestyle=linestyle,
                            label=label, markersize=8, linewidth=2)
    
    # Add ideal speedup line
    procs = sorted(df['mpi_procs'].unique())
    plt.plot(procs, procs, 'k--', label='Ideal Speedup', alpha=0.5, linewidth=2)
    
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Speedup (vs 1 MPI process)')
    plt.title('Speedup vs MPI Ranks')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('speedup_vs_mpi.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    
    print("="*60)
    print("MPI BENCHMARK ANALYSIS")
    print("="*60)
    
    # Load all results
    df = load_all_mpi_results()
    if df is None:
        return
    
    # Generate plots
    print("\n1. Plotting GFLOP/s vs MPI Processes...")
    plot_gflops_vs_mpi(df)
    
    print("\n2. Plotting Hybrid Scaling (MPI+OpenMP)...")
    plot_hybrid_scaling(df)
    
    print("\n3. Plotting Speedup vs MPI Ranks...")
    plot_speedup_vs_mpi(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated plots:")
    print("1. gflops_vs_mpi.png - GFLOP/s vs MPI Processes")
    print("2. hybrid_scaling.png - MPI+OpenMP Hybrid Scaling")
    print("3. speedup_vs_mpi.png - Speedup vs MPI Ranks")
    print("4. mpi_scaling.png - Basic MPI scaling (if hybrid not available)")
    print("="*60)

if __name__ == "__main__":
    # Simple styling
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    
    main()
