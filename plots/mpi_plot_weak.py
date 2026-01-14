import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load all CSV files
def load_data(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Load data
strong_data = load_data("result_MPI*.csv")
weak_data = load_data("weak_scaling_MPI*.csv") if glob.glob("weak_scaling_MPI*.csv") else None

# Calculate baselines for speedup
baselines = {}
for matrix in strong_data['matrix'].unique():
    matrix_data = strong_data[strong_data['matrix'] == matrix]
    baseline = matrix_data[matrix_data['mpi_procs'] == 1]
    if len(baseline) > 0:
        baselines[matrix] = baseline['avg_ms'].iloc[0]
    else:
        min_mpi = matrix_data['mpi_procs'].min()
        baselines[matrix] = matrix_data[matrix_data['mpi_procs'] == min_mpi]['avg_ms'].iloc[0]

matrix_colors = {}
matrices = list(strong_data['matrix'].unique())
colors = plt.cm.Set1(np.linspace(0, 1, len(matrices)))  # Different colors for each matrix
for i, matrix in enumerate(matrices):
    matrix_colors[matrix] = colors[i]

print(f"Color scheme:")
for matrix, color in matrix_colors.items():
    print(f"  {matrix}: {color}")

# GRAPH 1: Execution Time vs MPI Processes
plt.figure(figsize=(10, 6))
for matrix in strong_data['matrix'].unique():
    matrix_color = matrix_colors[matrix]
    matrix_data = strong_data[strong_data['matrix'] == matrix]
    
    # 1D partitioning - solid line
    part_data = matrix_data[matrix_data['partitioning'] == '1D']
    if len(part_data) > 0:
        times_by_mpi = part_data.groupby('mpi_procs')['avg_ms'].mean()
        plt.plot(times_by_mpi.index, times_by_mpi.values, 
                marker='o', linestyle='-', color=matrix_color,
                label=f'{matrix} - 1D')
    
    # 2D partitioning - dotted line
    part_data = matrix_data[matrix_data['partitioning'] == '2D']
    if len(part_data) > 0:
        times_by_mpi = part_data.groupby('mpi_procs')['avg_ms'].mean()
        plt.plot(times_by_mpi.index, times_by_mpi.values, 
                marker='s', linestyle=':', color=matrix_color,  # Dotted line for 2D
                label=f'{matrix} - 2D', linewidth=2)

plt.xlabel('MPI Processes')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time per SpMV')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('execution_time.png', dpi=100, bbox_inches='tight')
plt.show()

# GRAPH 2: Speedup per MPI Rank
plt.figure(figsize=(10, 6))
for matrix in strong_data['matrix'].unique():
    if matrix in baselines:
        matrix_color = matrix_colors[matrix]
        matrix_data = strong_data[strong_data['matrix'] == matrix]
        
        # 1D partitioning - solid line
        part_data = matrix_data[matrix_data['partitioning'] == '1D']
        if len(part_data) > 0:
            speedups = []
            mpi_values = []
            for mpi in sorted(part_data['mpi_procs'].unique()):
                mpi_data = part_data[part_data['mpi_procs'] == mpi]
                avg_time = mpi_data['avg_ms'].mean()
                speedup = baselines[matrix] / avg_time if avg_time > 0 else 0
                speedups.append(speedup)
                mpi_values.append(mpi)
            
            plt.plot(mpi_values, speedups, marker='o', linestyle='-',
                    color=matrix_color, label=f'{matrix} - 1D')
        
        # 2D partitioning - dotted line
        part_data = matrix_data[matrix_data['partitioning'] == '2D']
        if len(part_data) > 0:
            speedups = []
            mpi_values = []
            for mpi in sorted(part_data['mpi_procs'].unique()):
                mpi_data = part_data[part_data['mpi_procs'] == mpi]
                avg_time = mpi_data['avg_ms'].mean()
                speedup = baselines[matrix] / avg_time if avg_time > 0 else 0
                speedups.append(speedup)
                mpi_values.append(mpi)
            
            plt.plot(mpi_values, speedups, marker='s', linestyle=':', 
                    color=matrix_color, label=f'{matrix} - 2D', linewidth=2)

# Ideal speedup line
max_mpi = strong_data['mpi_procs'].max()
plt.plot([1, max_mpi], [1, max_mpi], 'k--', label='Ideal', alpha=0.5)

plt.xlabel('MPI Processes')
plt.ylabel('Speedup')
plt.title('Speedup per MPI Rank')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('speedup.png', dpi=100, bbox_inches='tight')
plt.show()

# GRAPH 3: Efficiency per Rank
plt.figure(figsize=(10, 6))
for matrix in strong_data['matrix'].unique():
    if matrix in baselines:
        matrix_color = matrix_colors[matrix]
        matrix_data = strong_data[strong_data['matrix'] == matrix]
        
        # 1D partitioning - solid line
        part_data = matrix_data[matrix_data['partitioning'] == '1D']
        if len(part_data) > 0:
            efficiencies = []
            mpi_values = []
            for mpi in sorted(part_data['mpi_procs'].unique()):
                mpi_data = part_data[part_data['mpi_procs'] == mpi]
                avg_time = mpi_data['avg_ms'].mean()
                speedup = baselines[matrix] / avg_time if avg_time > 0 else 0
                efficiency = speedup / mpi if mpi > 0 else 0
                efficiencies.append(efficiency)
                mpi_values.append(mpi)
            
            plt.plot(mpi_values, efficiencies, marker='o', linestyle='-',
                    color=matrix_color, label=f'{matrix} - 1D')
        
        # 2D partitioning - dotted line
        part_data = matrix_data[matrix_data['partitioning'] == '2D']
        if len(part_data) > 0:
            efficiencies = []
            mpi_values = []
            for mpi in sorted(part_data['mpi_procs'].unique()):
                mpi_data = part_data[part_data['mpi_procs'] == mpi]
                avg_time = mpi_data['avg_ms'].mean()
                speedup = baselines[matrix] / avg_time if avg_time > 0 else 0
                efficiency = speedup / mpi if mpi > 0 else 0
                efficiencies.append(efficiency)
                mpi_values.append(mpi)
            
            plt.plot(mpi_values, efficiencies, marker='s', linestyle=':',
                    color=matrix_color, label=f'{matrix} - 2D', linewidth=2)

plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal', alpha=0.5)
plt.xlabel('MPI Processes')
plt.ylabel('Efficiency per Rank')
plt.title('Efficiency per MPI Rank')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('efficiency.png', dpi=100, bbox_inches='tight')
plt.show()

# GRAPH 4: GFLOPs Performance
plt.figure(figsize=(10, 6))
for matrix in strong_data['matrix'].unique():
    matrix_color = matrix_colors[matrix]
    matrix_data = strong_data[strong_data['matrix'] == matrix]
    
    # 1D partitioning - solid line
    part_data = matrix_data[matrix_data['partitioning'] == '1D']
    if len(part_data) > 0:
        gflops_by_mpi = part_data.groupby('mpi_procs')['gflops'].mean()
        plt.plot(gflops_by_mpi.index, gflops_by_mpi.values, 
                marker='o', linestyle='-', color=matrix_color,
                label=f'{matrix} - 1D')
    
    # 2D partitioning - dotted line
    part_data = matrix_data[matrix_data['partitioning'] == '2D']
    if len(part_data) > 0:
        gflops_by_mpi = part_data.groupby('mpi_procs')['gflops'].mean()
        plt.plot(gflops_by_mpi.index, gflops_by_mpi.values, 
                marker='s', linestyle=':', color=matrix_color,
                label=f'{matrix} - 2D', linewidth=2)

plt.xlabel('MPI Processes')
plt.ylabel('GFLOPs')
plt.title('GFLOPs Performance')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('gflops.png', dpi=100, bbox_inches='tight')
plt.show()

# GRAPH 5: Communication vs Computation
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(2)  # For 1D and 2D

comm_data = {'1D': 0, '2D': 0}
comp_data = {'1D': 0, '2D': 0}

for partitioning in ['1D', '2D']:
    part_data = strong_data[strong_data['partitioning'] == partitioning]
    if len(part_data) > 0:
        avg_comm_ratio = (part_data['avg_comm_ms'].mean() / part_data['avg_ms'].mean() * 100) \
                         if part_data['avg_ms'].mean() > 0 else 0
        avg_comp_ratio = 100 - avg_comm_ratio
        
        comm_data[partitioning] = avg_comm_ratio
        comp_data[partitioning] = avg_comp_ratio

# Use consistent colors for bars too
plt.bar(index, [comm_data['1D'], comm_data['2D']], bar_width, 
        label='Communication', color='#FF6B6B')
plt.bar(index, [comp_data['1D'], comp_data['2D']], bar_width,
        bottom=[comm_data['1D'], comm_data['2D']], 
        label='Computation', color='#4ECDC4')

plt.xlabel('Partitioning Scheme')
plt.ylabel('Percentage (%)')
plt.title('Communication vs Computation Ratio')
plt.xticks(index, ['1D', '2D'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('comm_comp_ratio.png', dpi=100, bbox_inches='tight')
plt.show()

# GRAPH 6: Weak Scaling (if data exists)
if weak_data is not None and len(weak_data) > 0:
    plt.figure(figsize=(10, 6))
    # For weak scaling, use different markers for each partitioning
    markers = {'1D': 'o', '2D': 's'}
    
    for partitioning in ['1D', '2D']:
        part_data = weak_data[weak_data['partitioning'] == partitioning]
        if len(part_data) > 0:
            efficiencies = []
            mpi_values = sorted(part_data['mpi_procs'].unique())
            
            # Calculate weak scaling efficiency
            ref = part_data[part_data['mpi_procs'] == 1]
            if len(ref) > 0:
                ref_time = ref['avg_ms'].iloc[0]
                for mpi in mpi_values:
                    mpi_data = part_data[part_data['mpi_procs'] == mpi]
                    avg_time = mpi_data['avg_ms'].mean()
                    efficiency = ref_time / avg_time if avg_time > 0 else 0
                    efficiencies.append(efficiency)
                
                # Use dotted line for 2D partitioning
                linestyle = ':' if partitioning == '2D' else '-'
                plt.plot(mpi_values, efficiencies, marker=markers[partitioning], 
                        linestyle=linestyle, label=f'{partitioning}', linewidth=2)
    
    plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal', alpha=0.5)
    plt.xlabel('MPI Processes')
    plt.ylabel('Weak Scaling Efficiency')
    plt.title('Weak Scaling Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('weak_scaling.png', dpi=100, bbox_inches='tight')
    plt.show()

print("\nGraphs saved with:")
print("- Same color for each matrix across all graphs")
print("- Solid lines for 1D partitioning")
print("- Dotted lines (:) for 2D partitioning")
print("- Circle markers (o) for 1D")
print("- Square markers (s) for 2D")
print("\nSaved as:")
print("1. execution_time.png")
print("2. speedup.png")  
print("3. efficiency.png")
print("4. gflops.png")
print("5. comm_comp_ratio.png")
if weak_data is not None:
    print("6. weak_scaling.png")
