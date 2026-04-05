"""Data loading and cleaning helpers for the JD course project.

This module is intentionally simple and notebook-friendly. It focuses only on
reading the six raw CSV tables and applying light, defensive cleaning so the
rest of the project can start from consistent pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


TABLE_FILE_CANDIDATES = {
    "orders": ["JD_order_data.csv", "orders.csv", "order.csv"],
    "delivery": ["JD_delivery_data.csv", "delivery.csv"],
    "inventory": ["JD_inventory_data.csv", "inventory.csv"],
    "network": ["JD_network_data.csv", "network.csv"],
    "users": ["JD_user_data.csv", "users.csv", "user.csv"],
    "skus": ["JD_sku_data.csv", "skus.csv", "sku.csv"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, underscore-based column names."""
    out = df.copy()
    cleaned = []
    for col in out.columns:
        col_name = str(col).strip().lower()
        col_name = re.sub(r"[^a-z0-9]+", "_", col_name)
        col_name = re.sub(r"_+", "_", col_name).strip("_")
        cleaned.append(col_name)
    out.columns = cleaned
    return out


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from string-like columns and normalize blanks to NA."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def safe_to_datetime(series: pd.Series, format: str | None = None) -> pd.Series:
    """Safely parse a pandas Series to datetime, coercing invalid values to NaT."""
    if format is None:
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.to_datetime(series, format=format, errors="coerce")
    missing_mask = parsed.isna() & series.notna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(series.loc[missing_mask], errors="coerce")
    return parsed


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Safely parse a pandas Series to numeric, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


def find_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error if required columns are missing after normalization."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{table_name} is missing required columns: {missing_text}")


def _coerce_string_id_columns(df: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Keep ID-like fields as clean strings."""
    out = df.copy()
    id_columns = [col for col in out.columns if col.endswith("_id")]
    if extra_columns:
        id_columns.extend(extra_columns)
    for col in sorted(set(id_columns)):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def _coerce_integer_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Safely coerce selected columns to pandas nullable integer dtype."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = safe_to_numeric(out[col]).astype("Int64")
    return out


def _resolve_table_path(raw_dir: Path, table_name: str) -> Path:
    """Find the CSV path for a logical table name inside the raw data folder."""
    candidates = TABLE_FILE_CANDIDATES.get(table_name, [])
    for filename in candidates:
        path = raw_dir / filename
        if path.exists():
            return path
    candidate_text = ", ".join(candidates)
    raise ValueError(f"Could not find raw file for '{table_name}'. Expected one of: {candidate_text}")


def load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load and clean the six required JD raw tables from a directory.

    Parameters
    ----------
    raw_dir:
        Folder containing the raw CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with cleaned tables keyed by:
        `orders`, `delivery`, `inventory`, `network`, `users`, `skus`.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise ValueError(f"Raw data directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise ValueError(f"Raw data path is not a directory: {raw_path}")

    clean_functions = {
        "orders": clean_orders,
        "delivery": clean_delivery,
        "inventory": clean_inventory,
        "network": clean_network,
        "users": clean_users,
        "skus": clean_skus,
    }

    tables: dict[str, pd.DataFrame] = {}
    for table_name, cleaner in clean_functions.items():
        file_path = _resolve_table_path(raw_path, table_name)
        raw_df = pd.read_csv(file_path, dtype="string")
        tables[table_name] = cleaner(raw_df)

    return tables


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the orders table and create basic discount consistency fields."""
    out = standardize_columns(df)
    out = strip_string_columns(out)
    out = _coerce_string_id_columns(out)

    required = [
        "order_id",
        "user_id",
        "sku_id",
        "order_date",
        "order_time",
        "quantity",
        "type",
        "promise",
        "original_unit_price",
        "final_unit_price",
        "direct_discount_per_unit",
        "quantity_discount_per_unit",
        "bundle_discount_per_unit",
        "coupon_discount_per_unit",
        "gift_item",
        "dc_ori",
        "dc_des",
    ]
    find_required_columns(out, required, "orders")

    out["order_date"] = safe_to_datetime(out["order_date"]).dt.normalize()
    out["order_time"] = safe_to_datetime(out["order_time"])

    float_columns = [
        "original_unit_price",
        "final_unit_price",
        "direct_discount_per_unit",
        "quantity_discount_per_unit",
        "bundle_discount_per_unit",
        "coupon_discount_per_unit",
    ]
    for col in float_columns:
        out[col] = safe_to_numeric(out[col])

    out = _coerce_integer_columns(
        out,
        ["quantity", "type", "promise", "gift_item", "dc_ori", "dc_des"],
    )

    out["discount_total_per_unit"] = (
        out["direct_discount_per_unit"].fillna(0)
        + out["quantity_discount_per_unit"].fillna(0)
        + out["bundle_discount_per_unit"].fillna(0)
        + out["coupon_discount_per_unit"].fillna(0)
    )
    out["discount_gap"] = (
        out["original_unit_price"] - out["final_unit_price"] - out["discount_total_per_unit"]
    )

    consistent_mask = (
        out["original_unit_price"].notna()
        & out["final_unit_price"].notna()
        & out["discount_total_per_unit"].notna()
    )
    out["discount_consistency_flag"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[consistent_mask, "discount_consistency_flag"] = (
        out.loc[consistent_mask, "discount_gap"].abs() <= 1e-6
    ).astype("Int64")

    return out


def clean_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the delivery table and safely parse shipment timestamps."""
    out = standardize_columns(df)
    out = strip_string_columns(out)
    out = _coerce_string_id_columns(out)

    required = [
        "package_id",
        "order_id",
        "type",
        "ship_out_time",
        "arr_station_time",
        "arr_time",
    ]
    find_required_columns(out, required, "delivery")

    out = _coerce_integer_columns(out, ["type"])
    out["ship_out_time"] = safe_to_datetime(out["ship_out_time"])
    out["arr_station_time"] = safe_to_datetime(out["arr_station_time"])
    out["arr_time"] = safe_to_datetime(out["arr_time"])

    return out


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the inventory table, deduplicate, and add binary availability."""
    out = standardize_columns(df)
    out = strip_string_columns(out)
    out = _coerce_string_id_columns(out)

    required = ["dc_id", "sku_id", "date"]
    find_required_columns(out, required, "inventory")

    out["date"] = safe_to_datetime(out["date"]).dt.normalize()
    out = _coerce_integer_columns(out, ["dc_id"])
    out = out.drop_duplicates(subset=["date", "dc_id", "sku_id"]).reset_index(drop=True)
    out["inventory_available"] = 1

    return out


def clean_network(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and deduplicate the network table."""
    out = standardize_columns(df)
    out = strip_string_columns(out)

    required = ["region_id", "dc_id"]
    find_required_columns(out, required, "network")

    out = _coerce_integer_columns(out, ["region_id", "dc_id"])
    out = out.drop_duplicates(subset=["region_id", "dc_id"]).reset_index(drop=True)

    return out


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the users table and parse first order month when possible."""
    out = standardize_columns(df)
    out = strip_string_columns(out)
    out = _coerce_string_id_columns(out)

    required = ["user_id"]
    find_required_columns(out, required, "users")

    if "first_order_month" in out.columns:
        out["first_order_month"] = safe_to_datetime(out["first_order_month"], format="%Y-%m")

    return out


def clean_skus(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the SKU table and parse activation windows when available."""
    out = standardize_columns(df)
    out = strip_string_columns(out)
    out = _coerce_string_id_columns(out)

    required = ["sku_id"]
    find_required_columns(out, required, "skus")

    # Raw JD SKU extract can list the same sku_id more than once; merges expect m:1.
    out = out.drop_duplicates(subset=["sku_id"], keep="first").reset_index(drop=True)

    if "activate_date" in out.columns:
        out["activate_date"] = safe_to_datetime(out["activate_date"]).dt.normalize()
    if "deactivate_date" in out.columns:
        out["deactivate_date"] = safe_to_datetime(out["deactivate_date"]).dt.normalize()

    return out

