from __future__ import annotations

import pandas as pd


def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_direct_discount"] = (out["direct_discount_per_unit"] > 0).astype(int)
    out["has_quantity_discount"] = (out["quantity_discount_per_unit"] > 0).astype(int)
    out["has_bundle_discount"] = (out["bundle_discount_per_unit"] > 0).astype(int)
    out["has_coupon_discount"] = (out["coupon_discount_per_unit"] > 0).astype(int)
    out["any_discount_flag"] = (
        out[
            [
                "has_direct_discount",
                "has_quantity_discount",
                "has_bundle_discount",
                "has_coupon_discount",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)
    out["discount_depth"] = (
        (out["original_unit_price"] - out["final_unit_price"])
        / out["original_unit_price"].replace(0, pd.NA)
    )
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_hour"] = out["order_time"].dt.hour
    out["order_day"] = out["order_time"].dt.day
    out["order_weekday"] = out["order_time"].dt.dayofweek
    out["order_weekend_flag"] = out["order_weekday"].isin([5, 6]).astype(int)
    return out


def add_user_tenure_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"first_order_month", "order_date"}.issubset(out.columns):
        out["first_order_month"] = pd.to_datetime(out["first_order_month"], errors="coerce")
        out["user_tenure_days"] = (
            out["order_date"] - out["first_order_month"]
        ).dt.days.clip(lower=0)
    return out


def build_modeling_frame(base_df: pd.DataFrame) -> pd.DataFrame:
    df = add_promotion_features(base_df)
    df = add_time_features(df)
    df = add_user_tenure_feature(df)
    return df

