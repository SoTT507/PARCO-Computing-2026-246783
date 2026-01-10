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
    
    # 1. Speedup calculation (for strong scaling)
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
                    
                    # Efficiency per rank (MPI process)
                    efficiency_per_rank = speedup / row['mpi_procs'] if row['mpi_procs'] > 0 else 0
                    
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
                        'min_ms': row['min_ms'],
                        'max_ms': row['max_ms'],
                        'p90_ms': row['p90_ms'],
                        'avg_comm_ms': row['avg_comm_ms'],
                        'avg_comp_ms': row['avg_comp_ms'],
                        'gflops': row['gflops'],
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'efficiency_per_rank': efficiency_per_rank,
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
                avg_gflops = core_data['gflops'].mean()
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
                    'min_time_ms': core_data['min_ms'].mean(),
                    'max_time_ms': core_data['max_ms'].mean(),
                    'avg_gflops': avg_gflops,
                    'comm_ratio': avg_comm_ratio,
                    'weak_efficiency_time': weak_efficiency_time,
                    'weak_efficiency_gflops': weak_efficiency_gflops,
                    'nnz_per_proc': core_data['nnz'].mean() / total_cores if total_cores > 0 else 0
                })
    
    return pd.DataFrame(weak_analysis)

# ============================================
# REPORT GENERATION WITH PER-RANK METRICS
# ============================================

def generate_performance_report(metrics_df, weak_df=None):
    """
    Generate comprehensive performance report with per-rank metrics.
    """
    report_lines = []
    
    report_lines.append("=" * 120)
    report_lines.append("DISTRIBUTED SpMV PERFORMANCE EVALUATION REPORT")
    report_lines.append("=" * 120)
    
    # Overall statistics
    report_lines.append(f"\nOVERALL STATISTICS:")
    report_lines.append(f"- Total measurements: {len(metrics_df)}")
    report_lines.append(f"- Matrices analyzed: {', '.join(sorted(metrics_df['matrix'].unique()))}")
    report_lines.append(f"- Partitioning schemes: {list(metrics_df['partitioning'].unique())}")
    report_lines.append(f"- MPI configurations: {sorted(metrics_df['mpi_procs'].unique())}")
    report_lines.append(f"- OMP configurations: {sorted(metrics_df['omp_threads'].unique())}")
    
    # EXECUTION TIME PER SpMV - Detailed section
    report_lines.append("\n" + "=" * 120)
    report_lines.append("EXECUTION TIME PER SpMV (in milliseconds)")
    report_lines.append("=" * 120)
    
    for matrix in metrics_df['matrix'].unique():
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        report_lines.append(f"\nMatrix: {matrix}")
        report_lines.append("-" * 120)
        report_lines.append("Part | MPI | OMP |  Avg Time |  Min Time |  Max Time |  P90 Time | Time StdDev")
        report_lines.append("-" * 120)
        
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            for (mpi, omp), group in part_data.groupby(['mpi_procs', 'omp_threads']):
                avg_time = group['avg_ms'].mean()
                min_time = group['min_ms'].mean()
                max_time = group['max_ms'].mean()
                p90_time = group['p90_ms'].mean()
                time_std = group['avg_ms'].std() if len(group) > 1 else 0
                
                report_lines.append(f"{partitioning:4} | {mpi:3} | {omp:3} | "
                                  f"{avg_time:9.2f} | {min_time:9.2f} | "
                                  f"{max_time:9.2f} | {p90_time:9.2f} | {time_std:10.2f}")
    
    # SPEEDUP AND EFFICIENCY PER RANK - Detailed section
    report_lines.append("\n" + "=" * 120)
    report_lines.append("SPEEDUP AND EFFICIENCY (PER RANK)")
    report_lines.append("=" * 120)
    
    for matrix in metrics_df['matrix'].unique():
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        report_lines.append(f"\nMatrix: {matrix}")
        report_lines.append("-" * 120)
        report_lines.append("Part | MPI | OMP | Cores |  Speedup | Efficiency | Eff/Rank | MPI Eff | Comm %")
        report_lines.append("-" * 120)
        
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            for (mpi, omp), group in part_data.groupby(['mpi_procs', 'omp_threads']):
                avg_speedup = group['speedup'].mean()
                avg_efficiency = group['efficiency'].mean()
                avg_eff_per_rank = group['efficiency_per_rank'].mean()
                avg_mpi_eff = group['mpi_efficiency'].mean()
                avg_comm_ratio = group['comm_ratio'].mean() * 100
                total_cores = mpi * omp
                
                report_lines.append(f"{partitioning:4} | {mpi:3} | {omp:3} | "
                                  f"{total_cores:5} | {avg_speedup:8.2f} | "
                                  f"{avg_efficiency:10.2%} | {avg_eff_per_rank:8.2%} | "
                                  f"{avg_mpi_eff:7.2%} | {avg_comm_ratio:6.1f}")
    
    # Performance by partitioning summary
    report_lines.append("\n" + "=" * 120)
    report_lines.append("PERFORMANCE BY PARTITIONING SCHEME (SUMMARY)")
    report_lines.append("=" * 120)
    
    for partitioning in metrics_df['partitioning'].unique():
        part_data = metrics_df[metrics_df['partitioning'] == partitioning]
        report_lines.append(f"\n{partitioning} PARTITIONING:")
        
        # Average performance metrics
        avg_speedup = part_data['speedup'].mean()
        avg_efficiency = part_data['efficiency'].mean()
        avg_eff_per_rank = part_data['efficiency_per_rank'].mean()
        avg_gflops = part_data['gflops'].mean()
        avg_comm_ratio = part_data['comm_ratio'].mean() * 100
        
        report_lines.append(f"  Average Speedup: {avg_speedup:.2f}x")
        report_lines.append(f"  Average Efficiency: {avg_efficiency:.2%}")
        report_lines.append(f"  Average Efficiency per Rank: {avg_eff_per_rank:.2%}")
        report_lines.append(f"  Average GFLOPs: {avg_gflops:.4f}")
        report_lines.append(f"  Average Communication Ratio: {avg_comm_ratio:.1f}%")
        
        # Best configuration
        best_idx = part_data['gflops'].idxmax()
        best_row = part_data.loc[best_idx]
        report_lines.append(f"\n  Best Performance Configuration:")
        report_lines.append(f"    Matrix: {best_row['matrix']}")
        report_lines.append(f"    MPI Processes: {best_row['mpi_procs']}")
        report_lines.append(f"    OMP Threads: {best_row['omp_threads']}")
        report_lines.append(f"    Execution Time: {best_row['avg_ms']:.2f} ms")
        report_lines.append(f"    GFLOPs: {best_row['gflops']:.4f}")
        report_lines.append(f"    Speedup: {best_row['speedup']:.2f}x")
        report_lines.append(f"    Efficiency per Rank: {best_row['efficiency_per_rank']:.2%}")
    
    # Weak scaling analysis
    if weak_df is not None and len(weak_df) > 0:
        report_lines.append("\n" + "=" * 120)
        report_lines.append("WEAK SCALING ANALYSIS")
        report_lines.append("=" * 120)
        
        for partitioning in weak_df['partitioning'].unique():
            part_data = weak_df[weak_df['partitioning'] == partitioning]
            report_lines.append(f"\n{partitioning} PARTITIONING:")
            report_lines.append("-" * 120)
            report_lines.append("MPI | Avg Time(ms) | Min Time | Max Time | Weak Eff(Time) | Weak Eff(GFLOPs)")
            report_lines.append("-" * 120)
            
            for _, row in part_data.iterrows():
                report_lines.append(f"{row['mpi_procs']:3} | {row['avg_time_ms']:12.2f} | "
                                  f"{row['min_time_ms']:9.2f} | {row['max_time_ms']:9.2f} | "
                                  f"{row['weak_efficiency_time']:14.2%} | {row['weak_efficiency_gflops']:16.2%}")
    
    # Communication vs Computation breakdown
    report_lines.append("\n" + "=" * 120)
    report_lines.append("COMMUNICATION VS COMPUTATION BREAKDOWN")
    report_lines.append("=" * 120)
    
    for matrix in metrics_df['matrix'].unique():
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        report_lines.append(f"\nMatrix: {matrix}")
        report_lines.append("-" * 120)
        report_lines.append("Part | MPI | OMP | Comp Time(ms) | Comm Time(ms) | Comm % | Comp %")
        report_lines.append("-" * 120)
        
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            for (mpi, omp), group in part_data.groupby(['mpi_procs', 'omp_threads']):
                avg_comp = group['avg_comp_ms'].mean()
                avg_comm = group['avg_comm_ms'].mean()
                comm_percent = group['comm_ratio'].mean() * 100
                comp_percent = group['comp_ratio'].mean() * 100
                
                report_lines.append(f"{partitioning:4} | {mpi:3} | {omp:3} | "
                                  f"{avg_comp:13.2f} | {avg_comm:13.2f} | "
                                  f"{comm_percent:6.1f} | {comp_percent:6.1f}")
    
    # Memory footprint per rank
    report_lines.append("\n" + "=" * 120)
    report_lines.append("MEMORY FOOTPRINT PER RANK")
    report_lines.append("=" * 120)
    
    for matrix in metrics_df['matrix'].unique():
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        report_lines.append(f"\nMatrix: {matrix} (NNZ: {matrix_data['nnz'].iloc[0]:,})")
        report_lines.append("-" * 120)
        report_lines.append("Part | MPI | OMP | Mem/Rank(MB) | Mem/Core(MB)")
        report_lines.append("-" * 120)
        
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            for (mpi, omp), group in part_data.groupby(['mpi_procs', 'omp_threads']):
                avg_mem_per_rank = (group['nnz'].iloc[0] * 8 * 2) / (1024**2) / mpi
                avg_mem_per_core = group['memory_per_core_mb'].mean()
                
                report_lines.append(f"{partitioning:4} | {mpi:3} | {omp:3} | "
                                  f"{avg_mem_per_rank:12.2f} | {avg_mem_per_core:13.2f}")
    
    # Save report to file
    report_text = "\n".join(report_lines)
    with open("performance_report.txt", "w") as f:
        f.write(report_text)
    
    # Print key sections to console
    print("\n" + "=" * 80)
    print("KEY PERFORMANCE METRICS SUMMARY")
    print("=" * 80)
    
    print("\nEXECUTION TIME PER SpMV (ms):")
    print("-" * 80)
    print("Matrix           | Part | MPI | OMP | Avg Time | Speedup | Eff/Rank")
    print("-" * 80)
    for matrix in metrics_df['matrix'].unique():
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        for partitioning in matrix_data['partitioning'].unique():
            part_data = matrix_data[matrix_data['partitioning'] == partitioning]
            # Get the configuration with highest MPI count for comparison
            max_mpi = part_data['mpi_procs'].max()
            max_config = part_data[part_data['mpi_procs'] == max_mpi]
            if len(max_config) > 0:
                row = max_config.iloc[0]
                print(f"{matrix:16} | {partitioning:4} | {row['mpi_procs']:3} | "
                      f"{row['omp_threads']:3} | {row['avg_ms']:9.2f} | "
                      f"{row['speedup']:8.2f} | {row['efficiency_per_rank']:8.2%}")
    
    print(f"\nDetailed report saved to performance_report.txt ({len(report_lines)} lines)")
    
    return report_text

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
    
    # 1. Execution Time per SpMV
    ax1 = plt.subplot(3, 3, 1)
    for partitioning in metrics_df['partitioning'].unique():
        for matrix in metrics_df['matrix'].unique():
            subset = metrics_df[(metrics_df['partitioning'] == partitioning) & 
                               (metrics_df['matrix'] == matrix)]
            if len(subset) > 1:
                ax1.plot(subset['total_cores'], subset['avg_ms'], 
                        marker='o', linestyle='-', 
                        label=f'{matrix[:5]}-{partitioning}', alpha=0.7)
    ax1.set_xlabel('Total Cores (MPI × OMP)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time per SpMV')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup per Rank
    ax2 = plt.subplot(3, 3, 2)
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            speedup_by_mpi = subset.groupby('mpi_procs')['speedup'].mean()
            ax2.plot(speedup_by_mpi.index, speedup_by_mpi.values, 
                    marker='s', linestyle='-', linewidth=2, 
                    label=partitioning, alpha=0.8)
    ax2.plot(metrics_df['mpi_procs'].unique(), metrics_df['mpi_procs'].unique(), 
            'k--', label='Ideal', alpha=0.5)
    ax2.set_xlabel('MPI Processes')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup per MPI Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Efficiency per Rank
    ax3 = plt.subplot(3, 3, 3)
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            eff_by_mpi = subset.groupby('mpi_procs')['efficiency_per_rank'].mean()
            ax3.plot(eff_by_mpi.index, eff_by_mpi.values, 
                    marker='^', linestyle='-', linewidth=2,
                    label=partitioning, alpha=0.8)
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax3.set_xlabel('MPI Processes')
    ax3.set_ylabel('Efficiency per Rank')
    ax3.set_title('Efficiency per MPI Rank')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Communication vs Computation Breakdown
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
    
    # 5. Memory per Rank
    ax5 = plt.subplot(3, 3, 5)
    memory_data = []
    for partitioning in metrics_df['partitioning'].unique():
        for mpi_procs in sorted(metrics_df['mpi_procs'].unique()):
            subset = metrics_df[(metrics_df['partitioning'] == partitioning) & 
                               (metrics_df['mpi_procs'] == mpi_procs)]
            if len(subset) > 0:
                # Calculate memory per rank (not per core)
                avg_mem_per_rank = (subset['nnz'].iloc[0] * 8 * 2) / (1024**2) / mpi_procs
                memory_data.append({'Partitioning': partitioning, 
                                   'MPI Processes': mpi_procs,
                                   'Memory per Rank (MB)': avg_mem_per_rank})
    
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        for partitioning in memory_df['Partitioning'].unique():
            part_memory = memory_df[memory_df['Partitioning'] == partitioning]
            ax5.plot(part_memory['MPI Processes'], part_memory['Memory per Rank (MB)'],
                    marker='o', linestyle='-', label=partitioning)
    ax5.set_xlabel('MPI Processes')
    ax5.set_ylabel('Memory per Rank (MB)')
    ax5.set_title('Memory Footprint per Rank')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Weak Scaling Plot (if available)
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
    
    # 7. GFLOPs Performance
    ax7 = plt.subplot(3, 3, 7)
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            gflops_by_mpi = subset.groupby('mpi_procs')['gflops'].mean()
            ax7.plot(gflops_by_mpi.index, gflops_by_mpi.values, 
                    marker='o', linestyle='-', linewidth=2,
                    label=partitioning, alpha=0.8)
    ax7.set_xlabel('MPI Processes')
    ax7.set_ylabel('GFLOPs')
    ax7.set_title('GFLOPs Performance')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Execution Time Distribution
    ax8 = plt.subplot(3, 3, 8)
    time_data = []
    for partitioning in metrics_df['partitioning'].unique():
        subset = metrics_df[metrics_df['partitioning'] == partitioning]
        if len(subset) > 0:
            time_data.append({
                'Partitioning': partitioning,
                'Avg Time': subset['avg_ms'].mean(),
                'Min Time': subset['min_ms'].mean(),
                'Max Time': subset['max_ms'].mean()
            })
    
    if time_data:
        time_df = pd.DataFrame(time_data)
        x = np.arange(len(time_df))
        width = 0.25
        ax8.bar(x - width, time_df['Min Time'], width, label='Min Time', color='#2ecc71')
        ax8.bar(x, time_df['Avg Time'], width, label='Avg Time', color='#3498db')
        ax8.bar(x + width, time_df['Max Time'], width, label='Max Time', color='#e74c3c')
        ax8.set_xlabel('Partitioning')
        ax8.set_ylabel('Time (ms)')
        ax8.set_title('Execution Time Distribution')
        ax8.set_xticks(x)
        ax8.set_xticklabels(time_df['Partitioning'])
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Communication Time Analysis
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
