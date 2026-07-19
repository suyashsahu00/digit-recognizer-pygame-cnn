import matplotlib.pyplot as plt
import numpy as np

# Set academic stylesheet details
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

# Configure matplotlib parameters for professional appearance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cbd5e1' # Light slate grey borders
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.facecolor'] = 'white'

# Generate realistic mock training metrics over 15 epochs
epochs = np.arange(1, 16)

# Training accuracy asymptotically approaches 0.99
train_acc = 0.99 - 0.22 * np.exp(-0.25 * (epochs - 1))

# Validation accuracy approaches 0.985 with minor fluctuations
np.random.seed(42)  # For reproducible fluctuations
val_fluctuations = np.random.normal(0, 0.003, 15)
val_acc = 0.985 - 0.165 * np.exp(-0.28 * (epochs - 1)) + val_fluctuations
val_acc[-1] = 0.985  # Force exact endpoint matching requirement

# Loss curves mirroring the accuracy behavior
train_loss = 0.55 * np.exp(-0.27 * (epochs - 1)) + 0.015
val_loss = 0.48 * np.exp(-0.26 * (epochs - 1)) + 0.038 - (val_fluctuations * 1.5)
val_loss[-1] = 0.042  # Clean final validation loss

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Colors matching industrial/academic palettes
color_train = '#0f52ba'  # Sapphire Blue
color_val = '#dc2626'    # Red/Crimson

# 1. Plot Accuracy Curve
ax1.plot(epochs, train_acc, label='Training Accuracy', color=color_train, linewidth=2.0, marker='o', markersize=5)
ax1.plot(epochs, val_acc, label='Validation Accuracy', color=color_val, linewidth=2.0, linestyle='--', marker='s', markersize=5)
ax1.set_title('Model Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Epoch', fontsize=11, labelpad=8)
ax1.set_ylabel('Accuracy', fontsize=11, labelpad=8)
ax1.set_xticks(epochs)
ax1.set_ylim(0.75, 1.01)
ax1.grid(True, linestyle=':', color='#cbd5e1', alpha=0.8)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9, fontsize=10)

# 2. Plot Loss Curve
ax2.plot(epochs, train_loss, label='Training Loss', color=color_train, linewidth=2.0, marker='o', markersize=5)
ax2.plot(epochs, val_loss, label='Validation Loss', color=color_val, linewidth=2.0, linestyle='--', marker='s', markersize=5)
ax2.set_title('Model Loss vs. Epochs', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Epoch', fontsize=11, labelpad=8)
ax2.set_ylabel('Loss', fontsize=11, labelpad=8)
ax2.set_xticks(epochs)
ax2.set_ylim(-0.02, 0.62)
ax2.grid(True, linestyle=':', color='#cbd5e1', alpha=0.8)
ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9, fontsize=10)

# Adjust layouts and margins
plt.tight_layout()

# Save final graphic
output_path = 'training_graph.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated training curves graph: '{output_path}'")
