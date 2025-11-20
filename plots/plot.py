import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

csv_files = glob.glob("./*_results.csv")
if not csv_files:
    print("No CSV files found. Run benchmark first.")
    exit()

os.makedirs("./graphs", exist_ok=True)

all_data = []
for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

print(f"Loaded {len(df)} measurements from {df['matrix'].nunique()} matrices")

for matrix in df['matrix'].unique():
    matrix_data = df[df['matrix'] == matrix]
    csr_data = matrix_data[matrix_data['format'] == 'CSR']
    
    plt.figure(figsize=(10, 5))
    
    # Plot execution time
    for schedule in csr_data['schedule'].unique():
        sched_data = csr_data[csr_data['schedule'] == schedule]
        if not sched_data.empty:
            plt.plot(sched_data['threads'], sched_data['percentile_90'], 'o-', label=schedule)
    
    plt.xlabel('Threads')
    plt.ylabel('Time (ms)')
    plt.title(f'{matrix} - Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'benchmarks/plots/{matrix}_plot.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {matrix}_plot.png")

plt.figure(figsize=(12, 6))
for matrix in df['matrix'].unique():
    matrix_data = df[df['matrix'] == matrix]
    csr_data = matrix_data[matrix_data['format'] == 'CSR']
    
    # Get best performance (static schedule, max threads)
    best_data = csr_data[(csr_data['schedule'] == 'omp_static') & 
                         (csr_data['threads'] == csr_data['threads'].max())]
    if not best_data.empty:
        plt.bar(matrix, best_data['percentile_90'].iloc[0], alpha=0.7, label=matrix)

plt.xlabel('Matrix')
plt.ylabel('Best Time (ms)')
plt.title('Performance Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('benchmarks/plots/comparison.png', bbox_inches='tight')
plt.close()

print("Saved: comparison.png")
print("Done! Check benchmarks/plots/ directory")
