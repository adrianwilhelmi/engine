import pandas as pd
import matplotlib.pyplot as plt
import io
import re

data_str = """
Benchmark                                 Time             CPU   Iterations
BM_naive_slerp/repeats:10_mean          73.8 ns          73.5 ns             10
BM_naive_slerp/repeats:10_median        73.7 ns          73.3 ns             10
BM_naive_slerp/repeats:10_stddev        10.3 ns          10.1 ns             10
BM_naive_slerp/repeats:10_cv           14.02 %          13.81 %             10
BM_simd_slerp/repeats:10_mean           66.7 ns          66.0 ns             10
BM_simd_slerp/repeats:10_median         66.0 ns          65.6 ns             10
BM_simd_slerp/repeats:10_stddev         3.15 ns          3.03 ns             10
BM_simd_slerp/repeats:10_cv             4.72 %           4.60 %             10
BM_fast_slerp/repeats:10_mean           16.6 ns          16.4 ns             10
BM_fast_slerp/repeats:10_median         16.5 ns          16.4 ns             10
BM_fast_slerp/repeats:10_stddev        0.668 ns         0.533 ns             10
BM_fast_slerp/repeats:10_cv             4.04 %           3.24 %             10
"""

df = pd.read_csv(io.StringIO(data_str.strip()), sep=r'\s{2,}', engine='python')

def clean_val(val):
    if isinstance(val, str):
        return float(re.sub(r'[^\d.]', '', val))
    return val

df['CPU'] = df['CPU'].apply(clean_val)

libs = {
    'BM_naive_slerp': 'Naive C++ Slerp',
    'BM_simd_slerp': 'SIMD Slerp',
    'BM_fast_slerp': 'Fast Approx Slerp'
}

plot_data = []
for key, label in libs.items():
    mean_row = df[df['Benchmark'].str.contains(f"{key}.*_mean")]
    std_row = df[df['Benchmark'].str.contains(f"{key}.*_stddev")]
    
    if not mean_row.empty and not std_row.empty:
        mean = mean_row['CPU'].values[0]
        std = std_row['CPU'].values[0]
        plot_data.append({'Library': label, 'Mean': mean, 'StdDev': std})

pdf_df = pd.DataFrame(plot_data).sort_values('Mean', ascending=False)

plt.figure(figsize=(10, 6))
colors = ['#F44336', '#2196F3', '#4CAF50'] 
bars = plt.bar(pdf_df['Library'], pdf_df['Mean'], yerr=pdf_df['StdDev'], 
               capsize=10, color=colors, alpha=0.8, edgecolor='black')

plt.ylabel('Time [ns]', fontsize=12)
plt.title('Comparison of Quaternion Slerp Implementations', fontsize=14, pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.2f} ns', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('quat_slerp_bench_results.pdf')
plt.savefig('quat_slerp_bench_results.png')
