import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional output
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']

# 1. Generate realistic dummy confusion matrix data
np.random.seed(42)
num_classes = 10
matrix = np.zeros((num_classes, num_classes), dtype=int)
total_per_class = 100  # Total test samples per digit

for i in range(num_classes):
    # High accuracy diagonal element (between 88% and 97%)
    diag_val = np.random.randint(88, 98)
    matrix[i, i] = diag_val
    
    # Distribute the remaining samples among other classes (confusions)
    remaining = total_per_class - diag_val
    
    # Establish weights for realistic digit confusions
    weights = np.ones(num_classes)
    weights[i] = 0  # Cannot confuse with itself
    
    if i == 3:
        weights[8] = 4  # 3 is often confused with 8
    elif i == 8:
        weights[3] = 4  # 8 is often confused with 3
    elif i == 4:
        weights[9] = 5  # 4 is often confused with 9
    elif i == 9:
        weights[4] = 5  # 9 is often confused with 4
    elif i == 7:
        weights[2] = 3  # 7 is sometimes confused with 2
    elif i == 5:
        weights[6] = 3  # 5 is sometimes confused with 6
        
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Draw confusions using a multinomial distribution
    confusions = np.random.multinomial(remaining, weights)
    for j in range(num_classes):
        if j != i:
            matrix[i, j] = confusions[j]

# 2. Draw the Seaborn Heatmap
plt.figure(figsize=(10.5, 8.5), dpi=300)

# Customize the heatmap styling
sns.heatmap(
    matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=range(num_classes), 
    yticklabels=range(num_classes),
    linewidths=0.75,
    linecolor='#cbd5e1',  # Slate grid borders
    cbar_kws={'label': 'Sample Count', 'shrink': 0.82},
    annot_kws={'size': 11, 'weight': 'normal'}
)

# 3. Labeling and Aesthetics
plt.title('Confusion Matrix for MNIST Digit Classifier', fontsize=14, fontweight='bold', pad=20, color='#0f172a')
plt.xlabel('Predicted Digit Class', fontsize=12, labelpad=12, fontweight='bold', color='#1e293b')
plt.ylabel('True Digit Class', fontsize=12, labelpad=12, fontweight='bold', color='#1e293b')

# Rotate ticks for better alignment
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)

plt.tight_layout()

# 4. Save the generated diagram
output_path = 'confusion_matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated confusion matrix heatmap: '{output_path}'")
