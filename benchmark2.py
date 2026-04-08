import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Your benchmark data
epochs = np.arange(6)
cpp = np.array([117429, 129375, 128442, 128228, 128245, 128316])
torch = np.array([10373.7, 10675.0, 10458.5, 10612.3, 10426.8, 10169.5])

# Calculate Multiplier: C++ / Torch
multiplier = cpp / torch

# Setup Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

# 1. Heatmap (Multiplier style)
heatmap_df = pd.DataFrame(multiplier.reshape(1, -1), 
                          columns=[f'Ep {i}' for i in epochs],
                          index=['Speedup'])

sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap='Greens', ax=ax1, 
            annot_kws={'size': 14}, cbar=False)
# Add 'x' suffix to annotations manually for clarity
for text in ax1.texts: text.set_text(text.get_text() + "x")
ax1.set_title('Speedup Multiplier (C++ vs PyTorch)', fontweight='bold')

# 2. Line Graph (Multiplier style)
ax2.plot(epochs, multiplier, marker='o', color='green', linewidth=3)
ax2.set_ylabel('Multiplier (x times faster)')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 15) # Gives perspective
ax2.grid(True, alpha=0.3)
ax2.set_title('Performance Lead over Epochs')

plt.savefig('performance_multiplier.png', dpi=300, bbox_inches='tight')
plt.show()
