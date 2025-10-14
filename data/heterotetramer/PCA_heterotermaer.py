import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- Heterotetramer data ---
data = pd.DataFrame({
    'Shared': [3, 1, 2, 5],
    'Novel':  [4, 6, 2, 2],
    'Lost':   [0, 0, 1, 1]
}, index=[
    'ErbB1(2)–ErbB3(2)',
    'ErbB1(2)–ErbB4(2)',
    'ErbB2(2)–ErbB4(2)',
    'ErbB3(2)–ErbB4(2)'
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

# --- PC Scores ---
pc_scores = pd.DataFrame(
    scores,
    columns=[f"PC{i+1}_Score" for i in range(scores.shape[1])],
    index=data.index
)
print("\nPC Scores:\n", pc_scores)

# Combine raw + PC scores
combined = pd.concat([data, pc_scores], axis=1)
combined.to_csv("PCA_heterotetramer_scores.csv")
print("\nCombined data saved to 'PCA_heterotetramer_scores.csv'.")

# --- Correlation between PC1 and Novel motifs ---
r, p = pearsonr(pc_scores['PC1_Score'], data['Novel'])
print(f"\nPearson correlation (PC1 vs Novel): r = {r:.3f}, p = {p:.4f}")

# --- Plot ---
plt.figure(figsize=(6, 5))
plt.scatter(pc_scores['PC1_Score'], pc_scores['PC2_Score'], color='seagreen', s=80)

for i, label in enumerate(data.index):
    plt.text(pc_scores.iloc[i, 0] + 0.02,
             pc_scores.iloc[i, 1] + 0.02,
             label, fontsize=9)

plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
plt.title('PCA - Heterotetramer Interfaces')
plt.grid(True)
plt.tight_layout()

plt.savefig('PCA_heterotetramer_PC1_PC2.png', dpi=300)
plt.show()
