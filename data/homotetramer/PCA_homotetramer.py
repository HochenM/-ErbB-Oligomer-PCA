import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Homotetramer data ---
data = pd.DataFrame({
    'Shared': [2, 1, 1, 2],
    'Novel':  [1, 4, 7, 3],
    'Lost':   [1, 1, 1, 0]
}, index=[
    'ErbB1(2)–ErbB1(2)',
    'ErbB2(2)–ErbB2(2)',
    'ErbB3(2)–ErbB3(2)',
    'ErbB4(2)–ErbB4(2)'
])

# --- PCA ---
pca = PCA()
scores = pca.fit_transform(data)

# Variance explained
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
plt.scatter(scores[:, 0], scores[:, 1], color='tomato', s=80)

for i, label in enumerate(data.index):
    plt.text(scores[i, 0] + 0.02, scores[i, 1] + 0.02, label, fontsize=9)

plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA - Homotetramer Interfaces')
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig('PCA_homotetramer_PC1_PC2.png', dpi=300)
plt.show()
