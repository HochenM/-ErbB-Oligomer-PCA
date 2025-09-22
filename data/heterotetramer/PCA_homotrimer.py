import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Homotrimer data ---
data = pd.DataFrame({
    'Shared': [2, 1, 1, 2],
    'Novel':  [0, 1, 1, 3],
    'Lost':   [1, 1, 1, 1]
}, index=[
    'ErbB1–ErbB1–ErbB1',
    'ErbB2–ErbB2–ErbB2',
    'ErbB3–ErbB3–ErbB3',
    'ErbB4–ErbB4–ErbB4'
])

# --- PCA ---
pca = PCA()
scores = pca.fit_transform(data)

# Variance explained by each PC
explained_variance = pca.explained_variance_ratio_ * 100
print(f"Explained variance (%): PC1={explained_variance[0]:.2f}%, "
      f"PC2={explained_variance[1]:.2f}%, PC3={explained_variance[2]:.2f}%")

# Loadings
loadings = pd.DataFrame(
    pca.components_,
    columns=data.columns,
    index=[f"PC{i+1}" for i in range(len(data.columns))]
)
print("\nLoadings:\n", loadings)

# --- Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(scores[:, 0], scores[:, 1], color='dodgerblue', s=80)

# Labels for points
for i, label in enumerate(data.index):
    plt.text(scores[i, 0] + 0.02, scores[i, 1] + 0.02, label, fontsize=9)

plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA - Homotrimer Interfaces')
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig('PCA_homotrimer_PC1_PC2.png', dpi=300)
plt.show()
