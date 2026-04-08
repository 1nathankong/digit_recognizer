import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data
epochs = np.arange(6)
cpp = np.array([117429, 129375, 128442, 128228, 128245, 128316])
torch = np.array([10373.7, 10675.0, 10458.5, 10612.3, 10426.8, 10169.5])
speedup_pct = ((cpp - torch) / torch) * 100

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(epochs, cpp, marker='o', label='C++ (raw kernel code)', color='#3498db')
ax1.plot(epochs, torch, marker='s', label='PyTorch', color='#e74c3c')
ax1.set_yscale('log')
ax1.set_title('Raw Throughput Comparison')
ax1.legend()

heatmap_df = pd.DataFrame(speedup_pct.reshape(1, -1), index=['% Speedup'])
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap='RdYlGn', ax=ax2)
ax2.set_title('Percentage Efficiency Gain')

plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
