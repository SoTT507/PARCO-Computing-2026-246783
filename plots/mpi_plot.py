import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING AND PROCESSING
# ============================================

def load_performance_data(file_pattern="result_MPI*.csv"):
    """
    Load all performance CSV files matching the pattern.
    """
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['filename'] = os.path.basename(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded from files")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows from {len(files)} files")
    return combined_df

def load_weak_scaling_data(file_pattern="weak_scaling_MPI*.csv"):
    """
    Load weak scaling CSV files.
    """
    return load_performance_data(file_pattern)

# ============================================
# METRICS CALCULATION
# ============================================

def calculate_derived_metrics(df):
    """
    Calculate additional performance metrics from the loaded data.
    """
    df = df.copy()
    
    # Speedup calculation (for strong scaling)
    # Base case: single process performance for each matrix
    speedup_data = []
    
    for matrix in df['matrix'].unique():
        matrix_data = df[df['matrix'] == matrix]
        
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            
            # Find single process baseline (MPI=1, OMP=1 or lowest config)
            baseline = part_data[part_data['mpi_procs'] == 1]
            if len(baseline) == 0:
                # Use minimum MPI count available as baseline
                min_mpi = part_data['mpi_procs'].min()
                baseline = part_data[part_data['mpi_procs'] == min_mpi]
            
            if len(baseline) > 0:
                baseline_time = baseline['avg_ms'].iloc[0]
                
                for idx, row in part_data.iterrows():
                    # Speedup = baseline_time / current_time
                    speedup = baseline_time / row['avg_ms'] if row['avg_ms'] > 0 else 0
                    
                    # Efficiency = speedup / (mpi_procs * omp_threads)
                    total_cores = row['mpi_procs'] * row['omp_threads']
                    efficiency = speedup / total_cores if total_cores > 0 else 0
                    
                    # Parallel efficiency (just for MPI)
                    mpi_efficiency = speedup / row['mpi_procs'] if row['mpi_procs'] > 0 else 0
                    
                    # Communication ratio
                    comm_ratio = row['avg_comm_ms'] / row['avg_ms'] if row['avg_ms'] > 0 else 0
                    
                    # Computation ratio
                    comp_ratio = row['avg_comp_ms'] / row['avg_ms'] if row['avg_ms'] > 0 else 0
                    
                    # Memory per core estimation (simplified: 8 bytes per nnz)
                    memory_per_core_mb = (row['nnz'] * 8 * 2) / (1024**2) / total_cores  # x2 for indices
                    
                    # FLOPs per second (already in gflops, convert to MFLOPs for consistency)
                    mflops = row['gflops'] * 1000
                    
                    speedup_data.append({
                        'matrix': row['matrix'],
                        'partitioning': row['partitioning'],
                        'mpi_procs': row['mpi_procs'],
                        'omp_threads': row['omp_threads'],
                        'total_cores': total_cores,
                        'nnz': row['nnz'],
                        'avg_ms': row['avg_ms'],
                        'avg_comm_ms': row['avg_comm_ms'],
                        'avg_comp_ms': row['avg_comp_ms'],
                        'gflops': row['gflops'],  # lowercase as in CSV
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'mpi_efficiency': mpi_efficiency,
                        'comm_ratio': comm_ratio,
                        'comp_ratio': comp_ratio,
                        'memory_per_core_mb': memory_per_core_mb,
                        'mflops': mflops,
                        'filename': row.get('filename', '')
                    })
    
    metrics_df = pd.DataFrame(speedup_data)
    return metrics_df

def analyze_weak_scaling(df):
    """
    Analyze weak scaling performance.
    """
    weak_analysis = []
    
    for partitioning in df['partitioning'].unique():
        part_data = df[df['partitioning'] == partitioning]
        
        # Group by total core count (problem size should scale with cores)
        for total_cores in sorted(part_data['mpi_procs'].unique()):
            core_data = part_data[part_data['mpi_procs'] == total_cores]
            
            if len(core_data) > 0:
                avg_time = core_data['avg_ms'].mean()
                avg_gflops = core_data['gflops'].mean()  # lowercase as in CSV
                avg_comm_ratio = core_data['avg_comm_ms'].mean() / avg_time if avg_time > 0 else 0
                
                # Weak scaling efficiency: time should remain constant
                # Use first data point as reference
                if total_cores == part_data['mpi_procs'].min():
                    ref_time = avg_time
                    ref_gflops = avg_gflops
                    weak_efficiency_time = 1.0
                    weak_efficiency_gflops = 1.0
                else:
                    weak_efficiency_time = ref_time / avg_time if avg_time > 0 else 0
                    weak_efficiency_gflops = avg_gflops / ref_gflops if ref_gflops > 0 else 0
                
                weak_analysis.append({
                    'partitioning': partitioning,
                    'mpi_procs': total_cores,
                    'avg_time_ms': avg_time,
                    'avg_gflops': avg_gflops,  # lowercase as in CSV
                    'comm_ratio': avg_comm_ratio,
                    'weak_efficiency_time': weak_efficiency_time,
                    'weak_efficiency_gflops': weak_efficiency_gflops,
                    'nnz_per_proc': core_data['nnz'].mean() / total_cores if total_cores > 0 else 0
                })
    
    return pd.DataFrame(weak_analysis)

# ============================================
# VISUALIZATION - USING CORRECT COLUMN NAMES
# ============================================

def create_performance_plots(metrics_df, weak_df=None):
    """
    Create comprehensive performance visualizations.
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Strong Scaling: Speedup vs Cores
    ax1 = plt.subplot(3, 3, 1)
    for partitioning in metrics_df['partitioning'].unique():
        for matrix in metrics_df['matrix'].unique():
            subset = metrics_df[(metrics_df['partitioning'] == partitioning) & 
                               (metrics_df['matrix'] == matrix)]
            if len(subset) > 1:
                ax1.plot(subset['total_cores'], subset['speedup'], 
                        marker='o', linestyle='-', 
                        label=f'{matrix[:5]}-{partitioning}', alpha=0.7)
    ax1.plot(metrics_df['total_cores'], metrics_df['total_cores'], 
            'k--', label='Ideal', alpha=0.5)
    ax1.set_xlabel('Total Cores (MPI × OMP)')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling: Speedup')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Strong Scaling: Efficiency vs Cores
    ax2 = plt.subplot(3, 3, 2)
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            efficiency_by_cores = subset.groupby('total_cores')['efficiency'].mean()
            ax2.plot(efficiency_by_cores.index, efficiency_by_cores.values, 
                    marker='s', linestyle='-', linewidth=2, 
                    label=partitioning, alpha=0.8)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax2.set_xlabel('Total Cores (MPI × OMP)')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Strong Scaling: Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # GFLOPs vs Cores
    ax3 = plt.subplot(3, 3, 3)
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            gflops_by_cores = subset.groupby('total_cores')['gflops'].mean()  # lowercase
            ax3.plot(gflops_by_cores.index, gflops_by_cores.values, 
                    marker='^', linestyle='-', linewidth=2,
                    label=partitioning, alpha=0.8)
    ax3.set_xlabel('Total Cores (MPI × OMP)')
    ax3.set_ylabel('GFLOPs')
    ax3.set_title('Performance: GFLOPs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Communication vs Computation Breakdown
    ax4 = plt.subplot(3, 3, 4)
    comm_data = []
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            avg_comm = subset['comm_ratio'].mean() * 100
            avg_comp = subset['comp_ratio'].mean() * 100
            comm_data.append([partitioning, avg_comm, avg_comp])
    
    comm_df = pd.DataFrame(comm_data, columns=['Partitioning', 'Communication', 'Computation'])
    comm_df.plot(x='Partitioning', kind='bar', stacked=True, ax=ax4, 
                color=['#FF6B6B', '#4ECDC4'])
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Communication vs Computation Ratio')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.legend(['Communication', 'Computation'])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Memory per Core
    ax5 = plt.subplot(3, 3, 5)
    memory_data = []
    for partitioning in metrics_df['partitioning'].unique():
        for mpi_procs in sorted(metrics_df['mpi_procs'].unique()):
            subset = metrics_df[(metrics_df['partitioning'] == partitioning) & 
                               (metrics_df['mpi_procs'] == mpi_procs)]
            if len(subset) > 0:
                avg_memory = subset['memory_per_core_mb'].mean()
                memory_data.append({'Partitioning': partitioning, 
                                   'MPI Processes': mpi_procs,
                                   'Memory per Core (MB)': avg_memory})
    
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        for partitioning in memory_df['Partitioning'].unique():
            part_memory = memory_df[memory_df['Partitioning'] == partitioning]
            ax5.plot(part_memory['MPI Processes'], part_memory['Memory per Core (MB)'],
                    marker='o', linestyle='-', label=partitioning)
    ax5.set_xlabel('MPI Processes')
    ax5.set_ylabel('Memory per Core (MB)')
    ax5.set_title('Memory Footprint per Core')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Weak Scaling Plot (if available)
    if weak_df is not None and len(weak_df) > 0:
        ax6 = plt.subplot(3, 3, 6)
        for partitioning in weak_df['partitioning'].unique():
            subset = weak_df[weak_df['partitioning'] == partitioning]
            ax6.plot(subset['mpi_procs'], subset['weak_efficiency_gflops'],
                    marker='D', linestyle='-', linewidth=2,
                    label=f'{partitioning} (GFLOPs)', alpha=0.8)
            ax6.plot(subset['mpi_procs'], subset['weak_efficiency_time'],
                    marker='s', linestyle='--', linewidth=2,
                    label=f'{partitioning} (Time)', alpha=0.8)
        ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal')
        ax6.set_xlabel('MPI Processes')
        ax6.set_ylabel('Weak Scaling Efficiency')
        ax6.set_title('Weak Scaling Performance')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Heatmap: Performance by MPI × OMP configuration
    ax7 = plt.subplot(3, 3, 7)
    if len(metrics_df) > 0:
        # Create pivot table for heatmap
        pivot_data = metrics_df.pivot_table(
            values='gflops',  # lowercase
            index='mpi_procs',
            columns='omp_threads',
            aggfunc='mean'
        )
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax7)
            ax7.set_title('GFLOPs by MPI × OMP Configuration')
            ax7.set_xlabel('OMP Threads')
            ax7.set_ylabel('MPI Processes')
    
    # Comparison of 1D vs 2D partitioning - FIXED
    ax8 = plt.subplot(3, 3, 8)
    comparison_data = []
    for mpi_procs in sorted(metrics_df['mpi_procs'].unique()):
        for partitioning in ['1D', '2D']:
            subset = metrics_df[(metrics_df['partitioning'] == partitioning) & 
                               (metrics_df['mpi_procs'] == mpi_procs)]
            if len(subset) > 0:
                avg_gflops = subset['gflops'].mean()  # lowercase
                comparison_data.append({'MPI Processes': mpi_procs,
                                       'Partitioning': partitioning,
                                       'GFLOPs': avg_gflops})  # Capitalized for display

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        for partitioning in comparison_df['Partitioning'].unique():
            part_data = comparison_df[comparison_df['Partitioning'] == partitioning]
            # Using the correct column name from comparison_data
            ax8.plot(part_data['MPI Processes'], part_data['GFLOPs'],
                    marker='o', linestyle='-', label=partitioning)
        ax8.set_xlabel('MPI Processes')
        ax8.set_ylabel('Average GFLOPs')
        ax8.set_title('1D vs 2D Partitioning Comparison')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Communication Time Analysis
    ax9 = plt.subplot(3, 3, 9)
    comm_time_data = []
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            for mpi_procs in sorted(subset['mpi_procs'].unique()):
                mpi_subset = subset[subset['mpi_procs'] == mpi_procs]
                avg_comm = mpi_subset['avg_comm_ms'].mean()
                avg_total = mpi_subset['avg_ms'].mean()
                comm_time_data.append({'MPI Processes': mpi_procs,
                                      'Partitioning': partitioning,
                                      'Avg Comm Time (ms)': avg_comm,
                                      'Avg Total Time (ms)': avg_total})
    
    if comm_time_data:
        comm_time_df = pd.DataFrame(comm_time_data)
        for partitioning in comm_time_df['Partitioning'].unique():
            part_data = comm_time_df[comm_time_df['Partitioning'] == partitioning]
            ax9.plot(part_data['MPI Processes'], part_data['Avg Comm Time (ms)'],
                    marker='^', linestyle='-', label=f'{partitioning} Comm')
            ax9.plot(part_data['MPI Processes'], part_data['Avg Total Time (ms)'],
                    marker='o', linestyle='--', label=f'{partitioning} Total', alpha=0.5)
        ax9.set_xlabel('MPI Processes')
        ax9.set_ylabel('Time (ms)')
        ax9.set_title('Communication Time Analysis')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Distributed SpMV Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Performance plots saved to performance_analysis.png")

# ============================================
# REPORT GENERATION
# ============================================

def generate_performance_report(metrics_df, weak_df=None):
    """
    Generate comprehensive performance report.
    """
    report_lines = []
    
    report_lines.append("=" * 100)
    report_lines.append("DISTRIBUTED SpMV PERFORMANCE EVALUATION REPORT")
    report_lines.append("=" * 100)
    
    # Overall statistics
    report_lines.append(f"\nOVERALL STATISTICS:")
    report_lines.append(f"- Total measurements: {len(metrics_df)}")
    report_lines.append(f"- Matrices analyzed: {', '.join(sorted(metrics_df['matrix'].unique()))}")
    report_lines.append(f"- Partitioning schemes: {list(metrics_df['partitioning'].unique())}")
    report_lines.append(f"- MPI configurations: {sorted(metrics_df['mpi_procs'].unique())}")
    report_lines.append(f"- OMP configurations: {sorted(metrics_df['omp_threads'].unique())}")
    
    # Performance by partitioning
    report_lines.append("\n" + "=" * 100)
    report_lines.append("PERFORMANCE BY PARTITIONING SCHEME")
    report_lines.append("=" * 100)
    
    for partitioning in metrics_df['partitioning'].unique():
        part_data = metrics_df[metrics_df['partitioning'] == partitioning]
        report_lines.append(f"\n{partitioning} PARTITIONING:")
        
        # Average performance metrics
        avg_speedup = part_data['speedup'].mean()
        avg_efficiency = part_data['efficiency'].mean()
        avg_gflops = part_data['gflops'].mean()  # lowercase
        avg_comm_ratio = part_data['comm_ratio'].mean() * 100
        
        report_lines.append(f"  Average Speedup: {avg_speedup:.2f}x")
        report_lines.append(f"  Average Efficiency: {avg_efficiency:.2%}")
        report_lines.append(f"  Average GFLOPs: {avg_gflops:.4f}")
        report_lines.append(f"  Average Communication Ratio: {avg_comm_ratio:.1f}%")
        
        # Best configuration
        best_idx = part_data['gflops'].idxmax()  # lowercase
        best_row = part_data.loc[best_idx]
        report_lines.append(f"\n  Best Configuration:")
        report_lines.append(f"    MPI Processes: {best_row['mpi_procs']}")
        report_lines.append(f"    OMP Threads: {best_row['omp_threads']}")
        report_lines.append(f"    Matrix: {best_row['matrix']}")
        report_lines.append(f"    GFLOPs: {best_row['gflops']:.4f}")  # lowercase
        report_lines.append(f"    Speedup: {best_row['speedup']:.2f}x")
    
    # Weak scaling analysis
    if weak_df is not None and len(weak_df) > 0:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("WEAK SCALING ANALYSIS")
        report_lines.append("=" * 100)
        
        for partitioning in weak_df['partitioning'].unique():
            part_data = weak_df[weak_df['partitioning'] == partitioning]
            report_lines.append(f"\n{partitioning} PARTITIONING:")
            
            for _, row in part_data.iterrows():
                report_lines.append(f"\n  MPI Processes: {row['mpi_procs']}")
                report_lines.append(f"    Time: {row['avg_time_ms']:.2f} ms")
                report_lines.append(f"    Weak Scaling Efficiency (Time): {row['weak_efficiency_time']:.2%}")
                report_lines.append(f"    Weak Scaling Efficiency (GFLOPs): {row['weak_efficiency_gflops']:.2%}")
                report_lines.append(f"    Communication Ratio: {row['comm_ratio']:.1%}")
    
    # Detailed performance table
    report_lines.append("\n" + "=" * 100)
    report_lines.append("DETAILED PERFORMANCE METRICS")
    report_lines.append("=" * 100)
    report_lines.append("\nMatrix | Part | MPI | OMP | Cores | Time(ms) | GFLOPs | Speedup | Efficiency | Comm% | Mem/Core(MB)")
    report_lines.append("-" * 100)
    
    for _, row in metrics_df.iterrows():
        report_lines.append(f"{row['matrix'][:10]:10} | {row['partitioning']:4} | "
                          f"{row['mpi_procs']:3} | {row['omp_threads']:3} | "
                          f"{row['total_cores']:5} | {row['avg_ms']:8.2f} | "
                          f"{row['gflops']:7.4f} | {row['speedup']:8.2f} | "
                          f"{row['efficiency']:10.2%} | {row['comm_ratio']*100:5.1f} | "
                          f"{row['memory_per_core_mb']:12.2f}")
    
    # Save report to file
    report_text = "\n".join(report_lines)
    with open("performance_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text[:2000])  # Print first part of report
    print(f"\n... (truncated) ...")
    print(f"\nDetailed report saved to performance_report.txt")
    
    return report_text

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("Loading and analyzing SpMV performance data...")
    
    try:
        # Load strong scaling data
        strong_data = load_performance_data("result_MPI*.csv")
        metrics_df = calculate_derived_metrics(strong_data)
        
        # Load weak scaling data
        weak_data = None
        weak_analysis = None
        try:
            weak_data = load_weak_scaling_data("weak_scaling_MPI*.csv")
            weak_analysis = analyze_weak_scaling(weak_data)
        except FileNotFoundError:
            print("Warning: Weak scaling data files not found. Skipping weak scaling analysis.")
        
        # Generate reports
        report = generate_performance_report(metrics_df, weak_analysis)
        
        # Create visualizations
        create_performance_plots(metrics_df, weak_analysis)
        
        # Save processed data to CSV
        metrics_df.to_csv("processed_performance_metrics.csv", index=False)
        print("\nProcessed metrics saved to processed_performance_metrics.csv")
        
        if weak_analysis is not None:
            weak_analysis.to_csv("weak_scaling_analysis.csv", index=False)
            print("Weak scaling analysis saved to weak_scaling_analysis.csv")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"✓ Loaded {len(strong_data)} strong scaling measurements")
        if weak_data is not None:
            print(f"✓ Loaded {len(weak_data)} weak scaling measurements")
        print(f"✓ Generated performance report")
        print(f"✓ Created visualizations")
        print(f"✓ Saved processed data")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
