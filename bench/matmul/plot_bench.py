import pandas as pd
import matplotlib.pyplot as plt
import io

data_str = """name,iterations,real_time,cpu_time,time_unit,bytes_per_second,items_per_second,label,error_occurred,error_message
"BM_naive_matmul/repeats:10_mean",10,21.2177,20.8357,ns,,,,,
"BM_naive_matmul/repeats:10_stddev",10,3.17871,2.93188,ns,,,,,
"BM_custom_matmul/repeats:10_mean",10,13.8999,13.7549,ns,,,,,
"BM_custom_matmul/repeats:10_stddev",10,2.83215,2.71186,ns,,,,,
"BM_glm_matmul/repeats:10_mean",10,34.7393,34.497,ns,,,,,
"BM_glm_matmul/repeats:10_stddev",10,6.30218,6.18611,ns,,,,,"""

df = pd.read_csv(io.StringIO(data_str))

libs = {
    'BM_custom_matmul': 'my custom implementation',
    'BM_naive_matmul': 'naive C++ implementation',
    'BM_glm_matmul': 'GLM'
}

plot_data = []
for key, label in libs.items():
    mean = df[df['name'].str.contains(f"{key}.*_mean")]['cpu_time'].values[0]
    std = df[df['name'].str.contains(f"{key}.*_stddev")]['cpu_time'].values[0]
    plot_data.append({'Library': label, 'Mean': mean, 'StdDev': std})

pdf_df = pd.DataFrame(plot_data).sort_values('Mean')

plt.figure(figsize=(10, 6))
bars = plt.bar(pdf_df['Library'], pdf_df['Mean'], yerr=pdf_df['StdDev'], 
    capsize=10, color=['#4CAF50', '#2196F3', '#F44336'], alpha=0.8)

plt.ylabel('time [$ns$]', fontsize=12)
plt.title('Comparison of different 4x4 matrix matmul implementation', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
         f'{height:.2f} ns', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_results.pdf')
