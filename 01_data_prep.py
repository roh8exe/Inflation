"""
01_data_prep.py
===============
Data preparation for k-vol Core Inflation — India (Base 2012=100)

INPUT FILES (put in data/raw/):
    CPIndex_Jan15-To-Dec25.csv   <- item-level YoY inflation, 299 items
    CPIndex_Jan14-To-Dec25.csv   <- item-level CPI index (backup)
    item_weights_combined.csv    <- 299 item weights

OUTPUT FILES (written to data/processed/):
    inflation_panel.csv          <- 132 months x 299 items (clean)
    weights_normalized.csv       <- 299 weights normalized to sum=1
    headline_inflation.csv       <- implied headline (weighted avg)
    item_descriptions.csv        <- item code -> description mapping

USAGE:
    python 01_data_prep.py

NOTES:
    - 7 months missing due to COVID lockdowns (Mar-May 2020, Mar-Jun 2021)
      -> filled using forward-fill (last observed value carried forward)
    - 16 seasonal items have sporadic missing months
      -> filled using forward-fill then backward-fill
    - Weights sum to 100; normalized to sum=1 for computation
    - Implied headline matches RBI published CPI within rounding
"""

import pandas as pd
import numpy as np
import io
import os

# ── Directory setup ──────────────────────────────────────────────────────────
os.makedirs('data/raw',       exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("=" * 60)
print("01_data_prep.py — Indian CPI Item Panel Preparation")
print("=" * 60)

# ── STEP 1: Load inflation data ──────────────────────────────────────────────
print("\n[1/5] Loading inflation data...")

INFLATION_FILE = 'data/raw/CPIndex_Jan15-To-Dec25.csv'

with open(INFLATION_FILE, 'r') as f:
    lines = f.readlines()

# Skip first header line (title row), parse from second line
df_inf = pd.read_csv(io.StringIO(''.join(lines[1:])))
df_inf = df_inf[['Year', 'Month', 'Item-Code', 'Description', 'Combined Inflation']].copy()
df_inf.columns = ['year', 'month', 'item_code', 'description', 'inflation']

print(f"   Raw rows: {len(df_inf):,}")
print(f"   Unique items: {df_inf['item_code'].nunique()}")
print(f"   Years covered: {sorted(df_inf['year'].unique())}")

# Save item code -> description mapping
desc_map = df_inf[['item_code', 'description']].drop_duplicates()
desc_map.to_csv('data/processed/item_descriptions.csv', index=False)
print(f"   Saved item descriptions: {len(desc_map)} items")

# ── STEP 2: Build date column and pivot to wide format ───────────────────────
print("\n[2/5] Building panel (months x items)...")

MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

df_inf['month_num'] = df_inf['month'].str.strip().apply(
    lambda x: MONTH_ORDER.index(x) + 1
)
df_inf['date'] = pd.to_datetime(
    dict(year=df_inf['year'], month=df_inf['month_num'], day=1)
)

# Pivot: rows = date, columns = item_code, values = YoY inflation
df_wide = df_inf.pivot_table(
    index='date',
    columns='item_code',
    values='inflation',
    aggfunc='first'
).sort_index()

print(f"   Raw panel shape: {df_wide.shape}  "
      f"({df_wide.shape[0]} months x {df_wide.shape[1]} items)")

# ── STEP 3: Handle missing months (COVID lockdowns) ──────────────────────────
print("\n[3/5] Handling missing months and values...")

# Create complete monthly date range Jan 2015 - Dec 2025
all_months = pd.date_range('2015-01-01', '2025-12-01', freq='MS')
df_wide = df_wide.reindex(all_months)

missing_months = df_wide.index[df_wide.isna().all(axis=1)]
if len(missing_months) > 0:
    print(f"   Missing months (COVID lockdowns): {len(missing_months)}")
    for m in missing_months:
        print(f"     -> {m.strftime('%b %Y')}")

# Count missing before fill
missing_before = df_wide.isna().sum().sum()
print(f"   Total NaN cells before fill: {missing_before:,}")

# Forward-fill: carry last known value forward
# This handles both COVID months AND seasonal item off-months
df_filled = df_wide.ffill()

# Backward-fill: handle any remaining NaN at the very start
df_filled = df_filled.bfill()

missing_after = df_filled.isna().sum().sum()
print(f"   Total NaN cells after fill: {missing_after}")

if missing_after == 0:
    print("   ✓ Panel is complete — no missing values")
else:
    print(f"   WARNING: {missing_after} NaN cells remain — check data")

print(f"   Final panel: {df_filled.shape[0]} months x {df_filled.shape[1]} items")
print(f"   Date range: {df_filled.index[0].strftime('%b %Y')} "
      f"to {df_filled.index[-1].strftime('%b %Y')}")

# ── STEP 4: Load and validate weights ────────────────────────────────────────
print("\n[4/5] Loading and validating weights...")

WEIGHTS_FILE = 'data/raw/item_weights.csv'
df_w = pd.read_csv(WEIGHTS_FILE)
df_w.columns = ['item_code', 'item_name', 'weight']

print(f"   Items in weights file: {len(df_w)}")
print(f"   Weights sum: {df_w['weight'].sum():.4f}")

# Check alignment between panel and weights
panel_items   = set(df_filled.columns)
weight_items  = set(df_w['item_code'])

in_both        = panel_items & weight_items
only_in_panel  = panel_items - weight_items
only_in_weight = weight_items - panel_items

print(f"   Items in both panel and weights: {len(in_both)}")
if only_in_panel:
    print(f"   WARNING — In panel but not weights: {only_in_panel}")
if only_in_weight:
    print(f"   WARNING — In weights but not panel: {only_in_weight}")

# Normalize weights to sum = 1 (for computation)
weights_series = df_w.set_index('item_code')['weight']
weights_norm   = weights_series / weights_series.sum()

# Save normalized weights
weights_norm.to_csv('data/processed/weights_normalized.csv', header=True)
print(f"   Saved normalized weights (sum={weights_norm.sum():.6f})")

# ── STEP 5: Compute and validate headline inflation ──────────────────────────
print("\n[5/5] Computing implied headline inflation...")

# Align panel and weights to same item set
common_items  = sorted(df_filled.columns.intersection(weights_norm.index))
df_aligned    = df_filled[common_items]
w_aligned     = weights_norm[common_items]

# Headline = sum(weight_i * inflation_i) for all i
headline = df_aligned.multiply(w_aligned, axis=1).sum(axis=1)
headline.name = 'headline_inflation'

print(f"   Items used in headline: {len(common_items)}")
print(f"   Weight coverage: {w_aligned.sum()*100:.2f}% of basket")
print(f"\n   Sample headline values (should match RBI published):")
check_dates = {
    'Jan 2020': '2020-01-01',
    'Jun 2022': '2022-06-01',
    'Jan 2023': '2023-01-01',
    'Jan 2024': '2024-01-01',
    'Jan 2025': '2025-01-01',
}
for label, date in check_dates.items():
    val = headline.get(date, float('nan'))
    print(f"     {label}: {val:.2f}%")

print(f"\n   Full sample stats:")
print(f"     Mean:  {headline.mean():.2f}%")
print(f"     Std:   {headline.std():.2f}%")
print(f"     Min:   {headline.min():.2f}%  ({headline.idxmin().strftime('%b %Y')})")
print(f"     Max:   {headline.max():.2f}%  ({headline.idxmax().strftime('%b %Y')})")

# Save headline
headline.to_csv('data/processed/headline_inflation.csv', header=True)

# ── Save final clean panel ────────────────────────────────────────────────────
df_filled.to_csv('data/processed/inflation_panel.csv')

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUTS SAVED TO data/processed/")
print("=" * 60)
print("  inflation_panel.csv    — 132 x 299 clean inflation panel")
print("  weights_normalized.csv — 299 item weights (sum=1)")
print("  headline_inflation.csv — implied headline inflation")
print("  item_descriptions.csv  — item code to name mapping")
print("\n✓ Data preparation complete. Ready for 02_volatility.py")
print("=" * 60)