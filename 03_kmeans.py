"""
03_kmeans.py
============
K-means clustering of 299 CPI items by Historical Volatility.
This is the direct replication of Acosta (2018) for India.

WHAT THIS SCRIPT DOES:
    1. Finds optimal number of clusters K using Hartigan's Rule
    2. Runs k-means 500 times to build a probability matrix
       (each item gets a probability of belonging to each cluster)
    3. Orders clusters from lowest to highest volatility
    4. Saves the probability matrix for use in Script 04

WHY 500 RUNS?
    K-means is random — it starts from random initial points each time
    and may find different cluster assignments on different runs.
    Running 500 times and averaging gives stable, robust assignments.
    Example: if tomato is assigned to cluster 7 in 480/500 runs and
    cluster 8 in 20/500 runs, its probability vector is [0,0,...,0.96,0.04].

WHAT IS HARTIGAN'S RULE?
    A statistical test to find the right number of clusters K.
    We try K=1,2,3,...,15 and compute:
        H(K) = (SSE(K)/SSE(K+1) - 1) × (N - K - 1)
    where SSE = sum of squared distances within clusters.
    We stop at the first K where H(K) < 10.
    This is the same rule Acosta (2018) used — he got K=10 for Mexico.

INPUT:
    data/processed/item_volatility.csv   <- HV for 299 items

OUTPUT:
    data/processed/probability_matrix.csv  <- 299 x K matrix
    data/processed/cluster_summary.csv     <- cluster stats
    outputs/figures/hartigan_rule.png      <- K selection plot
    outputs/figures/cluster_hv_dist.png    <- HV distribution by cluster
    outputs/tables/cluster_items.txt       <- which items in which cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables',  exist_ok=True)

np.random.seed(42)

print("=" * 60)
print("03_kmeans.py — K-means Clustering on HV")
print("=" * 60)

# ── STEP 1: Load HV ──────────────────────────────────────────────────────────
print("\n[1/4] Loading Historical Volatility...")

HV = pd.read_csv('data/processed/item_volatility.csv',
                 index_col=0, header=0).squeeze()
HV.name = 'HV'
desc = pd.read_csv('data/processed/item_descriptions.csv',
                   index_col='item_code')['description']
weights = pd.read_csv('data/processed/weights_normalized.csv',
                      index_col=0).squeeze()

print(f"   Items: {len(HV)}")
print(f"   HV range: {HV.min():.2f} to {HV.max():.2f}")

# Reshape for sklearn — needs 2D array
X = HV.values.reshape(-1, 1)
N = len(X)

# ── STEP 2: Hartigan's Rule — find optimal K ─────────────────────────────────
print("\n[2/4] Finding optimal K using Hartigan's Rule...")
print("   (Testing K = 1 to 15)")

MAX_K   = 15
N_INIT  = 50    # runs per K for SSE computation (fast)

sse = {}
for k in range(1, MAX_K + 1):
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=42, max_iter=500)
    km.fit(X)
    sse[k] = km.inertia_   # inertia = SSE within clusters

print(f"\n   {'K':>3}  {'SSE':>12}  {'H statistic':>12}  {'Decision'}")
print(f"   {'-'*3}  {'-'*12}  {'-'*12}  {'-'*20}")

optimal_K = MAX_K
hartigan_stats = {}
for k in range(1, MAX_K):
    H = (sse[k] / sse[k + 1] - 1) * (N - k - 1)
    hartigan_stats[k] = H
    decision = ''
    if H < 10 and optimal_K == MAX_K:
        optimal_K = k
        decision  = '<-- STOP HERE (H < 10) = optimal K'
    print(f"   {k:>3}  {sse[k]:>12.2f}  {H:>12.2f}  {decision}")

print(f"\n   ✓ Optimal K = {optimal_K}")
print(f"   (Acosta 2018 found K=10 for Mexico; India may differ)")

# ── STEP 3: Run k-means 500 times — build probability matrix ─────────────────
print(f"\n[3/4] Running k-means 500 times with K={optimal_K}...")
print("   (This takes ~1-2 minutes)")

N_RUNS = 500
prob_matrix = np.zeros((N, optimal_K))

for run in range(N_RUNS):
    km = KMeans(n_clusters=optimal_K, n_init=1,
                random_state=run, max_iter=500)
    labels = km.fit_predict(X)

    # Re-order clusters by centroid: cluster 1 = lowest HV (most stable)
    centroids  = km.cluster_centers_.flatten()
    order      = np.argsort(centroids)          # ascending HV order
    rank_map   = {old: new for new, old in enumerate(order)}
    labels_ord = np.array([rank_map[l] for l in labels])

    # Add to probability matrix
    for i, lbl in enumerate(labels_ord):
        prob_matrix[i, lbl] += 1

# Normalize: each row sums to 1
prob_matrix = prob_matrix / N_RUNS

print(f"   ✓ Completed {N_RUNS} runs")

# Convert to DataFrame
cluster_cols = [f'cluster_{k+1}' for k in range(optimal_K)]
prob_df = pd.DataFrame(prob_matrix, index=HV.index, columns=cluster_cols)

# Most likely cluster for each item (for summary)
prob_df['modal_cluster'] = prob_df[cluster_cols].idxmax(axis=1)
prob_df['modal_prob']    = prob_df[cluster_cols].max(axis=1)

print(f"\n   Cluster assignment certainty (modal_prob):")
print(f"     Mean:   {prob_df['modal_prob'].mean():.3f}")
print(f"     Median: {prob_df['modal_prob'].median():.3f}")
print(f"     Min:    {prob_df['modal_prob'].min():.3f}  "
      f"(most uncertain item: "
      f"{desc.get(prob_df['modal_prob'].idxmin(), '?')[:40]})")

# ── STEP 4: Cluster summary ───────────────────────────────────────────────────
print(f"\n[4/4] Computing cluster summary...")

# Compute HV centroid per cluster (weighted by probability)
summary_rows = []
for k in range(optimal_K):
    col        = f'cluster_{k+1}'
    probs      = prob_df[col].values
    hv_centroid = (probs * HV.values).sum() / probs.sum()
    n_items    = (prob_df['modal_cluster'] == col).sum()
    basket_wt  = (probs * weights.values).sum() * 100  # % of basket

    summary_rows.append({
        'cluster':      k + 1,
        'hv_centroid':  hv_centroid,
        'n_items_modal': n_items,
        'basket_weight_pct': basket_wt
    })

summary = pd.DataFrame(summary_rows)

print(f"\n   {'Cluster':>8}  {'HV Centroid':>12}  "
      f"{'N Items':>8}  {'Basket Wt%':>12}")
print(f"   {'-'*8}  {'-'*12}  {'-'*8}  {'-'*12}")
for _, row in summary.iterrows():
    print(f"   {int(row.cluster):>8}  {row.hv_centroid:>12.2f}  "
          f"{int(row.n_items_modal):>8}  {row.basket_weight_pct:>12.2f}%")

# Save cluster summary
summary.to_csv('data/processed/cluster_summary.csv', index=False)

# Print which items are in each cluster
print(f"\n   Writing cluster membership to outputs/tables/cluster_items.txt")
with open('outputs/tables/cluster_items.txt', 'w') as f:
    f.write(f"K-vol Cluster Membership (K={optimal_K})\n")
    f.write(f"India CPI, Base 2012=100, Jan 2015 - Dec 2025\n")
    f.write("=" * 70 + "\n\n")
    for k in range(optimal_K):
        col       = f'cluster_{k+1}'
        hv_c      = summary.loc[summary.cluster == k+1, 'hv_centroid'].values[0]
        members   = prob_df[prob_df['modal_cluster'] == col].index
        f.write(f"CLUSTER {k+1}  (HV centroid = {hv_c:.2f})\n")
        f.write("-" * 50 + "\n")
        for code in members:
            name  = desc.get(code, code)
            hv    = HV.get(code, np.nan)
            w     = weights.get(code, 0) * 100
            p     = prob_df.loc[code, col]
            f.write(f"  [{p:.2f}] {name:<45} HV={hv:.1f}  w={w:.3f}%\n")
        f.write(f"  ({len(members)} items)\n\n")

# ── Save probability matrix ───────────────────────────────────────────────────
prob_df.to_csv('data/processed/probability_matrix.csv')
print(f"   Probability matrix saved: {prob_df.shape}")

# ── FIGURE 1: Hartigan's Rule plot ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ks = list(range(1, MAX_K))
hs = [hartigan_stats[k] for k in ks]

axes[0].plot(ks, hs, 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].axhline(y=10, color='red', linestyle='--', linewidth=1.5,
                label='H = 10 threshold')
axes[0].axvline(x=optimal_K, color='green', linestyle='--', linewidth=1.5,
                label=f'Optimal K = {optimal_K}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel("Hartigan's H Statistic", fontsize=12)
axes[0].set_title("Hartigan's Rule — Optimal K Selection\n"
                  "(India CPI 299 items)", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(ks)

# SSE elbow
axes[1].plot(list(sse.keys()), list(sse.values()), 's-',
             color='crimson', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_K, color='green', linestyle='--', linewidth=1.5,
                label=f'K = {optimal_K}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Within-Cluster SSE', fontsize=12)
axes[1].set_title('SSE Elbow Plot', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xticks(list(sse.keys()))

plt.tight_layout()
plt.savefig('outputs/figures/hartigan_rule.png', dpi=150, bbox_inches='tight')
plt.close()

# ── FIGURE 2: HV distribution by cluster ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, optimal_K))

for k in range(optimal_K):
    col     = f'cluster_{k+1}'
    members = prob_df[prob_df['modal_cluster'] == col].index
    hv_vals = HV[members].values
    hv_c    = summary.loc[summary.cluster == k+1, 'hv_centroid'].values[0]
    ax.scatter(
        hv_vals,
        [k + 1] * len(hv_vals),
        color=colors[k],
        s=60, alpha=0.7,
        label=f'C{k+1} (centroid={hv_c:.1f})'
    )

ax.set_xlabel('Historical Volatility (std dev of YoY inflation %)', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title(f'HV Distribution by Cluster (K={optimal_K})\n'
             f'India CPI 299 Items, Jan 2015 – Dec 2025', fontsize=12)
ax.set_yticks(range(1, optimal_K + 1))
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/cluster_hv_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Final output ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUTS SAVED")
print("=" * 60)
print(f"  Optimal K found         : {optimal_K}")
print(f"  data/processed/probability_matrix.csv")
print(f"  data/processed/cluster_summary.csv")
print(f"  outputs/tables/cluster_items.txt")
print(f"  outputs/figures/hartigan_rule.png")
print(f"  outputs/figures/cluster_hv_dist.png")
print(f"\n✓ Clustering complete. Ready for 04_build_cores.py")
print("=" * 60)