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
