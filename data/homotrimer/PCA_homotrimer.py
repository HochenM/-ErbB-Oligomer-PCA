import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
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

# --- PCA Scores Table ---
pc_scores = pd.DataFrame(
    scores,
    columns=[f"PC{i+1}_Score" for i in range(scores.shape[1])],
    index=data.index
)
print("\nPC Scores:\n", pc_scores)

# Combine raw data + PC scores
combined = pd.concat([data, pc_scores], axis=1)
combined.to_csv("PCA_homotrimer_scores.csv")
print("\nCombined data saved to 'PCA_homotrimer_scores.csv'.")

# --- Correlation between PC1 and Novel motifs ---
r, p = pearsonr(pc_scores['PC1_Score'], data['Novel'])
print(f"\nPearson correlation (PC1 vs Novel): r = {r:.3f}, p = {p:.4f}")

# --- Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(pc_scores['PC1_Score'], pc_scores['PC2_Score'], color='dodgerblue', s=80)

# Labels for points
for i, label in enumerate(data.index):
    plt.text(pc_scores.iloc[i, 0] + 0.02,
             pc_scores.iloc[i, 1] + 0.02,
             label, fontsize=9)

plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA - Homotrimer Interfaces')
plt.grid(True)
plt.tight_layout()

plt.savefig('PCA_homotrimer_PC1_PC2.png', dpi=300)
plt.show()
