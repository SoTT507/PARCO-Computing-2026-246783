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
    # parallel_df = df[df['schedule'].str.startswith('omp_')]

    #Uncomment above to print execution times across every scheduling strategy

    parallel_df = df[df['schedule']=='omp_guided']
    # GRAPH 1: Execution times
    plt.figure(figsize=(12, 6))
    
    for matrix in parallel_df['matrix'].unique():
        matrix_data = parallel_df[parallel_df['matrix'] == matrix]
        avg_times = matrix_data.groupby('threads')['percentile_90'].mean()
        plt.plot(avg_times.index, avg_times.values, 'o-', label=matrix, markersize=6, linewidth=2)
    
    plt.xlabel('Threads')
    plt.ylabel('Time (ms)')
    plt.title('90th Percentile Execution Time - Guided Scheduling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(parallel_df['threads'].unique()))
    
    plt.tight_layout()
    plt.savefig('execution_times.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # GRAPH 2: Speedup comparison
    plt.figure(figsize=(12, 7))
    
    # Get OMP Guided results
    guided_df = df[df['schedule'] == 'omp_guided']
    
    seq_times_dict = {}
    for matrix in df['matrix'].unique():
        seq_data = df[(df['matrix'] == matrix) & (df['schedule'] == 'sequential')]
        if not seq_data.empty:
            seq_times_dict[matrix] = seq_data['percentile_90'].iloc[0]
    
    for matrix in guided_df['matrix'].unique():
        matrix_data = guided_df[guided_df['matrix'] == matrix]
        seq_time = seq_times_dict.get(matrix)
        
        if seq_time is not None:
            speedup = seq_time / matrix_data['percentile_90']
            plt.plot(matrix_data['threads'], speedup, 's-', label=matrix, markersize=8, linewidth=2)
    
    # Ideal speedup line
    threads = sorted(guided_df['threads'].unique())
    plt.plot(threads, threads, 'k--', label='Ideal Speedup', alpha=0.5, linewidth=2)
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (vs Sequential)')
    plt.title('Speedup Improvement - OMP Guided vs Sequential')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(threads)

    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = load_results()
    if df is not None:
        plot_90th_percentile(df)
        print("Two separate graphs saved:")
        print("- 'execution_times.png' - Execution time comparison")
        print("- 'speedup_comparison.png' - Speedup comparison")
