import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score, roc_curve
)

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SAVE_DPI = 150


# ══════════════════════════════════════════════════════════════════════════════
# 0.  LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

orders   = pd.read_csv("JD_order_data.csv")
delivery = pd.read_csv("JD_delivery_data.csv")
sku      = pd.read_csv("JD_sku_data.csv")
users    = pd.read_csv("JD_user_data.csv")
inventory= pd.read_csv("JD_inventory_data.csv")
network  = pd.read_csv("JD_network_data.csv")

print(f"orders:    {orders.shape}")
print(f"delivery:  {delivery.shape}")
print(f"sku:       {sku.shape}")
print(f"users:     {users.shape}")
print(f"inventory: {inventory.shape}")
print(f"network:   {network.shape}")

# ── parse delivery timestamps ──────────────────────────────────────────────────
for col in ["ship_out_time", "arr_station_time", "arr_time"]:
    delivery[col] = pd.to_datetime(delivery[col], errors="coerce")

delivery["actual_delivery_hours"] = (
    delivery["arr_time"] - delivery["ship_out_time"]
).dt.total_seconds() / 3600

# ── merge master ───────────────────────────────────────────────────────────────
df = (orders
      .merge(sku[["sku_ID","type","brand_ID","attribute1","attribute2"]],
             on="sku_ID", how="left", suffixes=("","_sku"))
      .merge(users[["user_ID","user_level","plus","city_level","purchase_power"]],
             on="user_ID", how="left")
      .merge(delivery[["order_ID","actual_delivery_hours"]],
             on="order_ID", how="left"))

# ── rename conflicting type columns ───────────────────────────────────────────
# orders.type already exists; sku.type merged as type_sku
# We keep orders.type as "order_type" and sku.type as "product_type"
df.rename(columns={"type": "order_type", "type_sku": "product_type"}, inplace=True)

# ── engineered features ────────────────────────────────────────────────────────
df["discount_rate"] = np.where(
    df["original_unit_price"] > 0,
    (df["original_unit_price"] - df["final_unit_price"]) / df["original_unit_price"],
    0
)
df["discount_rate"] = df["discount_rate"].clip(0, 1)
df["is_discounted"]  = (df["discount_rate"] > 0).astype(int)
df["is_cross_dc"]    = (df["dc_ori"] != df["dc_des"]).astype(int)

# promise is in days (string with '-' for missing); convert to hours
df["promise"] = pd.to_numeric(df["promise"], errors="coerce") * 24
df["delivery_gap"]   = df["actual_delivery_hours"] - df["promise"]
df["late_delivery"]  = (df["delivery_gap"] > 0).astype(int)

# dominant discount type label
disc_cols = ["direct_discount_per_unit","quantity_discount_per_unit",
             "bundle_discount_per_unit","coupon_discount_per_unit"]
df["dominant_discount"] = df[disc_cols].apply(
    lambda r: disc_cols[r.values.argmax()].replace("_per_unit","")
    if r.max() > 0 else "no_discount", axis=1
)

# inventory availability at destination dc
orders["order_date_only"] = pd.to_datetime(orders["order_date"]).dt.date.astype(str)
inventory["date"] = inventory["date"].astype(str)

inv_key = inventory[["dc_ID","sku_ID","date"]].copy()
inv_key["inv_available"] = 1

df_temp = df.copy()
df_temp["order_date_only"] = pd.to_datetime(df_temp["order_date"]).dt.date.astype(str)
df_temp["dc_des_int"] = df_temp["dc_des"].astype("Int64")

df = df_temp.merge(
    inv_key.rename(columns={"dc_ID":"dc_des_int","sku_ID":"sku_ID","date":"order_date_only"}),
    on=["dc_des_int","sku_ID","order_date_only"], how="left"
)
df["inv_available"] = df["inv_available"].fillna(0).astype(int)

# region of destination dc
df = df.merge(network.rename(columns={"dc_ID":"dc_des_int"}),
              on="dc_des_int", how="left")

print(f"\nMaster dataframe shape: {df.shape}")
print(f"Missing actual_delivery_hours: {df['actual_delivery_hours'].isna().sum()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — FULFILLMENT PERFORMANCE FACTORS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: FULFILLMENT PERFORMANCE FACTORS")
print("=" * 60)
 
df_del = df.dropna(subset=["actual_delivery_hours"]).copy()
df_del = df_del[df_del["actual_delivery_hours"].between(0, 200)]  # sanity filter
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analysis 2 — Fulfillment Performance Factors", fontsize=15, fontweight="bold")
 
# ── 2a. Cross-DC vs Same-DC delivery hours ─────────────────────────────────────
ax = axes[0, 0]
cross   = df_del[df_del["is_cross_dc"] == 1]["actual_delivery_hours"]
same_dc = df_del[df_del["is_cross_dc"] == 0]["actual_delivery_hours"]
 
ax.hist(same_dc, bins=50, alpha=0.6, color="#4C72B0", label=f"Same DC (n={len(same_dc):,})",
        density=True)
ax.hist(cross,   bins=50, alpha=0.6, color="#DD8452", label=f"Cross DC (n={len(cross):,})",
        density=True)
ax.axvline(same_dc.median(), color="#4C72B0", linestyle="--", linewidth=1.5)
ax.axvline(cross.median(),   color="#DD8452", linestyle="--", linewidth=1.5)
ax.set_xlabel("Delivery Hours")
ax.set_ylabel("Density")
ax.set_title("(2a) Same-DC vs Cross-DC Delivery Time")
ax.legend()

plt.show()
 
u_stat, u_p = stats.mannwhitneyu(same_dc, cross, alternative="less")
print(f"\n[2a] Cross-DC delays delivery?")
print(f"  Same DC median = {same_dc.median():.1f}h,  Cross DC median = {cross.median():.1f}h")
print(f"  MWU test (same < cross): p = {u_p:.4e}  → {'Yes, significant' if u_p < 0.05 else 'Not significant'}")
 
# ── 2b. 1P vs 3P fulfillment hours ────────────────────────────────────────────
ax = axes[0, 1]
df_del_typed = df_del[df_del["product_type"].isin([1, 2])].copy()
df_del_typed["product_label"] = df_del_typed["product_type"].map({1:"1P","2":"3P"})
df_del_typed["product_label"] = df_del_typed["product_type"].map({1:"1P (JD Own)",2:"3P (Third-Party)"})
 
sns.violinplot(data=df_del_typed, x="product_label", y="actual_delivery_hours",
               palette={"1P (JD Own)":"#4C72B0","3P (Third-Party)":"#DD8452"},
               inner="quartile", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Delivery Hours")
ax.set_title("(2b) Delivery Time: 1P vs 3P")

plt.show()
 
p1d = df_del_typed[df_del_typed["product_type"]==1]["actual_delivery_hours"]
p3d = df_del_typed[df_del_typed["product_type"]==2]["actual_delivery_hours"]
_, p_type = stats.mannwhitneyu(p1d, p3d, alternative="two-sided")
print(f"\n[2b] 1P vs 3P delivery hours")
print(f"  1P median = {p1d.median():.1f}h,  3P median = {p3d.median():.1f}h")
print(f"  MWU p = {p_type:.4e}")
 
# ── 2c. Promised vs Actual delivery hours ─────────────────────────────────────
ax = axes[1, 0]
df_gap = df_del.dropna(subset=["promise", "delivery_gap"])
gap_counts = df_gap["late_delivery"].value_counts().rename({0:"On Time",1:"Late"})
 
ax.bar(gap_counts.index, gap_counts.values,
       color=["#4C72B0","#DD8452"], edgecolor="white", width=0.4)
for i, v in enumerate(gap_counts.values):
    ax.text(i, v + 500, f"{v:,}\n({v/gap_counts.sum()*100:.1f}%)", ha="center", fontsize=10)
ax.set_xticks(range(len(gap_counts)))
ax.set_xticklabels(gap_counts.index)
ax.set_ylabel("Number of Orders")
ax.set_title("(2c) On-Time vs Late Delivery")

plt.show()
 
print(f"\n[2c] Promised vs actual delivery")
print(f"  Late delivery rate = {df_gap['late_delivery'].mean()*100:.1f}%")
print(f"  Mean delivery gap  = {df_gap['delivery_gap'].mean():.1f}h")
 
# ── 2d. Inventory availability at destination DC ───────────────────────────────
ax = axes[1, 1]
inv_perf = (df_del.groupby("inv_available")["actual_delivery_hours"]
            .agg(["median","mean","count"]).reset_index())
inv_perf["label"] = inv_perf["inv_available"].map({0:"No Inv at Des DC",1:"Inv Available"})
 
ax.bar(inv_perf["label"], inv_perf["median"],
       color=["#DD8452","#4C72B0"], edgecolor="white", width=0.4)
for _, row in inv_perf.iterrows():
    ax.text(row["label"], row["median"] + 0.3,
            f'Median: {row["median"]:.1f}h\n(n={int(row["count"]):,})',
            ha="center", fontsize=9)
ax.set_ylabel("Median Delivery Hours")
ax.set_title("(2d) Delivery Time by Inventory Availability at Dest. DC")
 
plt.show()
 
inv0 = df_del[df_del["inv_available"]==0]["actual_delivery_hours"]
inv1 = df_del[df_del["inv_available"]==1]["actual_delivery_hours"]
if len(inv0) > 0 and len(inv1) > 0:
    _, p_inv = stats.mannwhitneyu(inv0, inv1, alternative="greater")
    print(f"\n[2d] Inventory availability")
    print(f"  No inv median  = {inv0.median():.1f}h,  Inv available median = {inv1.median():.1f}h")
    print(f"  MWU p (no_inv > inv) = {p_inv:.4e}")
 
plt.tight_layout()
