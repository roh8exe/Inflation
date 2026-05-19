"""
02_volatility.py
================
Compute clustering attributes for all 299 CPI items.

This is where the actual research begins.

WHAT THIS SCRIPT DOES:
    For each of the 299 items, computes 4 attributes:
    1. Historical Volatility (HV)       — std dev of inflation over full sample
    2. Seasonality Strength (SS)        — how seasonal the item is (STL decomp)
    3. Oil-Price Sensitivity (OS)       — correlation with global oil prices
    4. Monetary Sensitivity (MS)        — correlation with RBI repo rate changes

    Attribute 1 alone = Acosta (2018) replication
    Attributes 1+2+3+4 = our novel multi-attribute extension

    For Phase 1 (replication), only HV is used.
    For Phase 2 (extension), all 4 attributes are used.

INPUT FILES:
    data/processed/inflation_panel.csv      <- 132 x 299 clean panel
    data/processed/weights_normalized.csv   <- 299 item weights

OUTPUT FILES:
    data/processed/item_volatility.csv      <- HV for all 299 items
    data/processed/item_attributes.csv      <- all 4 attributes
    outputs/figures/volatility_dist.png     <- histogram of HV distribution
    outputs/figures/top_bottom_items.png    <- most/least volatile items

USAGE:
    python 02_volatility.py

NOTE ON OIL AND REPO DATA:
    If data/raw/brent_oil.csv and data/raw/repo_rate.csv are not yet
    available, attributes 3 and 4 will be skipped and a warning printed.
    This allows Phase 1 to proceed without external data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables',  exist_ok=True)

print("=" * 60)
print("02_volatility.py — Item Attribute Computation")
print("=" * 60)

# ── STEP 1: Load the clean panel ────────────────────────────────────────────
print("\n[1/5] Loading clean inflation panel...")

df = pd.read_csv('data/processed/inflation_panel.csv', index_col=0, parse_dates=True)
weights = pd.read_csv(
    'data/processed/weights_normalized.csv',
    index_col=0
).squeeze("columns")
desc = pd.read_csv('data/processed/item_descriptions.csv',
                   index_col='item_code')['description']

print(f"   Panel: {df.shape[0]} months x {df.shape[1]} items")
print(f"   Date range: {df.index[0].strftime('%b %Y')} to {df.index[-1].strftime('%b %Y')}")
print(f"   Weights: {len(weights)} items")

# ── STEP 2: Historical Volatility (HV) ──────────────────────────────────────
# This is the ONLY attribute used in Acosta (2018).
# HV = standard deviation of an item's full-sample inflation series.
# Items with high HV are volatile (tomato, onion, leechi).
# Items with low HV are stable (education fees, rent, medicine).
print("\n[2/5] Computing Historical Volatility (HV)...")

HV = df.std(axis=0)   # std dev across all 132 months, for each of 299 items
HV.name = 'HV'

print(f"   HV computed for {len(HV)} items")
print(f"   HV range: {HV.min():.2f} to {HV.max():.2f} percentage points")
print(f"   HV mean:  {HV.mean():.2f}")
print(f"   HV median:{HV.median():.2f}")

# Most and least volatile items
top10 = HV.nlargest(10)
bot10 = HV.nsmallest(10)

print(f"\n   TOP 10 MOST VOLATILE ITEMS:")
for code, val in top10.items():
    name = desc.get(code, code)
    w    = weights.get(code, 0) * 100
    print(f"     {val:6.2f}  {name:<45} (weight={w:.3f}%)")

print(f"\n   TOP 10 MOST STABLE ITEMS:")
for code, val in bot10.items():
    name = desc.get(code, code)
    w    = weights.get(code, 0) * 100
    print(f"     {val:6.2f}  {name:<45} (weight={w:.3f}%)")

# ── STEP 3: Seasonality Strength (SS) ───────────────────────────────────────
# Measures how much of an item's variation is seasonal vs random.
# Formula: SS = max(0, 1 - Var(residual) / Var(seasonal + residual))
# SS close to 1 = highly seasonal (mango, vegetables)
# SS close to 0 = non-seasonal (rent, medicine, education)
print("\n[3/5] Computing Seasonality Strength (SS)...")

try:
    from statsmodels.tsa.seasonal import STL

    seasonality = {}
    failed = []
    for i, item in enumerate(df.columns):
        if (i + 1) % 50 == 0:
            print(f"   Processing item {i+1}/{len(df.columns)}...")
        series = df[item].dropna()
        if len(series) < 24:   # need at least 2 years for STL
            seasonality[item] = np.nan
            failed.append(item)
            continue
        try:
            stl = STL(series, period=12, robust=True)
            res = stl.fit()
            var_resid    = res.resid.var()
            var_seasonal = res.seasonal.var()
            var_combined = (res.seasonal + res.resid).var()
            # Strength of seasonality (Wang et al. 2006 formula)
            ss = max(0.0, 1.0 - var_resid / var_combined) if var_combined > 0 else 0.0
            seasonality[item] = ss
        except Exception:
            seasonality[item] = np.nan
            failed.append(item)

    SS = pd.Series(seasonality, name='SS')
    print(f"   SS computed for {SS.notna().sum()} items "
          f"({len(failed)} failed/skipped)")
    print(f"   SS range: {SS.min():.3f} to {SS.max():.3f}")
    print(f"\n   TOP 10 MOST SEASONAL ITEMS:")
    for code, val in SS.nlargest(10).items():
        name = desc.get(code, code)
        print(f"     {val:.3f}  {name}")
    print(f"\n   TOP 10 LEAST SEASONAL ITEMS:")
    for code, val in SS.nsmallest(10).items():
        name = desc.get(code, code)
        print(f"     {val:.3f}  {name}")

except ImportError:
    print("   WARNING: statsmodels not available. SS set to NaN.")
    SS = pd.Series(np.nan, index=df.columns, name='SS')

# ── STEP 4: Oil-Price Sensitivity (OS) ──────────────────────────────────────
# Correlation of each item's inflation with global oil price YoY change.
# High OS = item tracks oil prices (petrol, diesel, kerosene, LPG)
# Low OS  = item independent of oil (rent, education, haircut)
print("\n[4/5] Computing Oil-Price Sensitivity (OS)...")

OIL_FILE = 'data/raw/brent_oil.csv'
if os.path.exists(OIL_FILE):
    oil = pd.read_csv(OIL_FILE, index_col=0, parse_dates=True)
    oil_yoy = oil.iloc[:, 0].pct_change(12) * 100
    oil_aligned = oil_yoy.reindex(df.index)
    valid_mask = oil_aligned.notna()
    OS = df.loc[valid_mask].corrwith(
        oil_aligned[valid_mask], axis=0
    )
    OS.name = 'OS'
    print(f"   OS computed for {OS.notna().sum()} items")
    print(f"   Most oil-sensitive: {OS.nlargest(5).index.tolist()}")
else:
    print("   INFO: data/raw/brent_oil.csv not found.")
    print("   OS set to NaN. Download from RBI DBIE for Phase 2.")
    OS = pd.Series(np.nan, index=df.columns, name='OS')

# ── STEP 5: Monetary Sensitivity (MS) ───────────────────────────────────────
# Correlation with RBI repo rate monthly changes.
# High MS = item responds to monetary policy (credit-sensitive goods)
# Low MS  = item insensitive to rate changes (food, fuel)
print("\n[5/5] Computing Monetary Sensitivity (MS)...")

REPO_FILE = 'data/raw/repo_rate.csv'
if os.path.exists(REPO_FILE):
    repo = pd.read_csv(REPO_FILE, index_col=0, parse_dates=True)
    repo_change = repo.iloc[:, 0].diff()
    repo_aligned = repo_change.reindex(df.index)
    valid_mask = repo_aligned.notna()
    MS = df.loc[valid_mask].corrwith(
        repo_aligned[valid_mask], axis=0
    )
    MS.name = 'MS'
    print(f"   MS computed for {MS.notna().sum()} items")
else:
    print("   INFO: data/raw/repo_rate.csv not found.")
    print("   MS set to NaN. Download from RBI DBIE for Phase 2.")
    MS = pd.Series(np.nan, index=df.columns, name='MS')

# ── Combine all attributes ───────────────────────────────────────────────────
attrs = pd.DataFrame({
    'HV': HV,
    'SS': SS,
    'OS': OS,
    'MS': MS,
    'weight': weights * 100,      # weight as percentage for readability
    'description': desc
})
attrs.index.name = 'item_code'

# Save
HV.to_csv('data/processed/item_volatility.csv', header=True)
attrs.to_csv('data/processed/item_attributes.csv')

print(f"\n   Attribute summary:")
print(attrs[['HV','SS','OS','MS']].describe().round(3).to_string())

# ── FIGURE 1: Volatility distribution ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of HV
axes[0].hist(HV.values, bins=40, color='steelblue', edgecolor='white',
             alpha=0.85)
axes[0].axvline(HV.median(), color='red', linestyle='--', linewidth=1.5,
                label=f'Median = {HV.median():.1f}')
axes[0].axvline(HV.mean(),   color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean = {HV.mean():.1f}')
axes[0].set_xlabel('Historical Volatility (std dev of YoY inflation, %)',
                   fontsize=11)
axes[0].set_ylabel('Number of Items', fontsize=11)
axes[0].set_title('Distribution of Item Volatility\n(299 CPI Items, India, Jan 2015–Dec 2025)',
                  fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Right: weight vs volatility scatter
sc = axes[1].scatter(
    HV.values,
    weights.values * 100,
    alpha=0.5,
    s=20,
    c=HV.values,
    cmap='RdYlGn_r'
)
plt.colorbar(sc, ax=axes[1], label='HV')

# Label a few important items
highlight = {
    'tomato': '1.1.07.3.1.01.0',
    'onion':  '1.1.07.1.1.02.0',
    'milk':   '1.1.04.1.1.01.X',
    'rent':   '4.1.01.1.2.01.X',
    'petrol': '6.1.03.2.1.01.0',
    'medicine': '6.1.02.2.1.01.X',
}
for label, code in highlight.items():
    if code in HV.index and code in weights.index:
        axes[1].annotate(
            label,
            xy=(HV[code], weights[code] * 100),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='darkred'
        )

axes[1].set_xlabel('Historical Volatility (HV)', fontsize=11)
axes[1].set_ylabel('Item Weight (% of basket)', fontsize=11)
axes[1].set_title('Volatility vs Basket Weight\n(bubble = item)',
                  fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/volatility_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n   Figure saved: outputs/figures/volatility_dist.png")

# ── FIGURE 2: Top/bottom volatile items bar chart ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Most volatile
top15 = HV.nlargest(15)
names_top = [desc.get(c, c)[:30] for c in top15.index]
axes[0].barh(range(len(top15)), top15.values[::-1],
             color='crimson', alpha=0.8)
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(names_top[::-1], fontsize=9)
axes[0].set_xlabel('Historical Volatility (std dev %)', fontsize=11)
axes[0].set_title('15 Most Volatile CPI Items\n(India, 2015–2025)', fontsize=12)
axes[0].grid(axis='x', alpha=0.3)

# Least volatile
bot15 = HV.nsmallest(15)
names_bot = [desc.get(c, c)[:30] for c in bot15.index]
axes[1].barh(range(len(bot15)), bot15.values,
             color='steelblue', alpha=0.8)
axes[1].set_yticks(range(len(bot15)))
axes[1].set_yticklabels(names_bot, fontsize=9)
axes[1].set_xlabel('Historical Volatility (std dev %)', fontsize=11)
axes[1].set_title('15 Most Stable CPI Items\n(India, 2015–2025)', fontsize=12)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/top_bottom_items.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Figure saved: outputs/figures/top_bottom_items.png")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUTS SAVED")
print("=" * 60)
print("  data/processed/item_volatility.csv  — HV for 299 items")
print("  data/processed/item_attributes.csv  — all 4 attributes")
print("  outputs/figures/volatility_dist.png")
print("  outputs/figures/top_bottom_items.png")

hv_ready = HV.notna().sum()
ss_ready = SS.notna().sum()
os_ready = OS.notna().sum()
ms_ready = MS.notna().sum()

print(f"\n  Attribute readiness:")
print(f"    HV (volatility)       : {hv_ready}/299 items ✓  — Phase 1 ready")
print(f"    SS (seasonality)      : {ss_ready}/299 items    — Phase 2")
print(f"    OS (oil sensitivity)  : {os_ready}/299 items    — Phase 2 (needs oil data)")
print(f"    MS (monetary)         : {ms_ready}/299 items    — Phase 2 (needs repo data)")

if hv_ready == 299:
    print("\n✓ HV complete. Ready for 03_kmeans.py")
print("=" * 60)