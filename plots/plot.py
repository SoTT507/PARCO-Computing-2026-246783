import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Simple setup
plt.rcParams['figure.figsize'] = (12, 6)

def load_results():
    """Load all CSV files"""
    csv_files = glob.glob("*_results.csv")
    if not csv_files:
        print("No CSV files found! Run benchmark first.")
        return None
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def plot_90th_percentile(df):
    """Simple 90th percentile comparison"""
    # Get parallel results only
    parallel_df = df[df['schedule'].str.startswith('omp_')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Execution times
    for matrix in parallel_df['matrix'].unique():
        matrix_data = parallel_df[parallel_df['matrix'] == matrix]
        avg_times = matrix_data.groupby('threads')['percentile_90'].mean()
        ax1.plot(avg_times.index, avg_times.values, 'o-', label=matrix, markersize=4)
    
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('90th Percentile Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup - FIXED VERSION
    # Get sequential times as a dictionary
    seq_times_dict = {}
    for matrix in df['matrix'].unique():
        seq_data = df[(df['matrix'] == matrix) & (df['schedule'] == 'sequential')]
        if not seq_data.empty:
            seq_times_dict[matrix] = seq_data['percentile_90'].iloc[0]
    
    for matrix in parallel_df['matrix'].unique():
        matrix_data = parallel_df[parallel_df['matrix'] == matrix]
        seq_time = seq_times_dict.get(matrix)  # Now it's a single value or None
        
        if seq_time is not None:  # FIX: Check if not None instead of truthiness
            avg_times = matrix_data.groupby('threads')['percentile_90'].mean()
            speedup = seq_time / avg_times
            ax2.plot(speedup.index, speedup.values, 's-', label=matrix, markersize=4)
    
    # Ideal speedup line
    threads = sorted(parallel_df['threads'].unique())
    ax2.plot(threads, threads, 'k--', label='Ideal', alpha=0.5)
    
    ax2.set_xlabel('Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Sequential')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('90th_percentile.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = load_results()
    if df is not None:
        plot_90th_percentile(df)
        print("90th percentile plot saved as '90th_percentile.png'")
