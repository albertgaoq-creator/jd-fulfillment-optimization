from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import JD_FILENAMES, RAW_DATA_DIR

DATE_COLUMNS = {
    "orders": ["order_date", "order_time"],
    "delivery": ["ship_out_time", "arr_station_time", "arr_time"],
    "inventory": ["date"],
    "skus": ["activate_date", "deactivate_date"],
    "users": ["first_order_month"],
}


def load_table(table_name: str, raw_dir: Path | str | None = None) -> pd.DataFrame:
    if table_name not in JD_FILENAMES:
        raise KeyError(f"Unknown table: {table_name}")

    base_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    path = base_dir / JD_FILENAMES[table_name]
    df = pd.read_csv(path)

    for col in DATE_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if table_name == "orders":
        if "promise" in df.columns:
            df["promise_days"] = pd.to_numeric(df["promise"], errors="coerce")
        discount_cols = [
            "direct_discount_per_unit",
            "quantity_discount_per_unit",
            "bundle_discount_per_unit",
            "coupon_discount_per_unit",
        ]
        for col in discount_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_core_tables(raw_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    table_names = ["orders", "delivery", "inventory", "network", "users", "skus"]
    return {name: load_table(name, raw_dir=raw_dir) for name in table_names}


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_share": df.isna().mean(),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report.sort_values(["missing_share", "missing_count"], ascending=False)


def duplicate_report(df: pd.DataFrame, key_cols: list[str]) -> dict[str, int]:
    duplicated_rows = int(df.duplicated(subset=key_cols).sum())
    unique_keys = int(df[key_cols].drop_duplicates().shape[0])
    return {
        "row_count": int(df.shape[0]),
        "unique_key_count": unique_keys,
        "duplicate_rows_on_key": duplicated_rows,
    }


def add_order_value_fields(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["gross_merchandise_value"] = df["quantity"] * df["original_unit_price"]
    df["net_merchandise_value"] = df["quantity"] * df["final_unit_price"]
    df["discount_total"] = df["gross_merchandise_value"] - df["net_merchandise_value"]
    df["discount_rate"] = df["discount_total"] / df["gross_merchandise_value"].replace(0, pd.NA)
    df["remote_fulfillment_flag"] = (df["dc_ori"] != df["dc_des"]).astype(int)
    df["gift_item_flag"] = df["gift_item"].fillna(0).astype(int)
    return df


def build_order_line_base(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    skus: pd.DataFrame,
) -> pd.DataFrame:
    df = add_order_value_fields(orders)
    df = df.merge(users, how="left", on="user_ID", validate="m:1")
    df = df.merge(skus, how="left", on="sku_ID", validate="m:1", suffixes=("", "_sku"))
    df["order_line_id"] = df["order_ID"].astype(str) + "_" + df["sku_ID"].astype(str)
    return df


def build_delivery_summary(delivery: pd.DataFrame) -> pd.DataFrame:
    df = delivery.copy()
    df["ship_to_arrival_hours"] = (
        (df["arr_time"] - df["ship_out_time"]).dt.total_seconds() / 3600.0
    )
    df["station_to_arrival_hours"] = (
        (df["arr_time"] - df["arr_station_time"]).dt.total_seconds() / 3600.0
    )
    summary = (
        df.groupby("order_ID", as_index=False)
        .agg(
            package_count=("package_ID", "nunique"),
            first_ship_out_time=("ship_out_time", "min"),
            final_arrival_time=("arr_time", "max"),
            mean_ship_to_arrival_hours=("ship_to_arrival_hours", "mean"),
        )
    )
    return summary


def build_order_fulfillment_fact(
    orders: pd.DataFrame,
    delivery: pd.DataFrame,
    users: pd.DataFrame,
    skus: pd.DataFrame,
) -> pd.DataFrame:
    order_lines = build_order_line_base(orders, users, skus)
    delivery_summary = build_delivery_summary(delivery)
    df = order_lines.merge(delivery_summary, how="left", on="order_ID", validate="m:1")
    df["hours_to_ship"] = (
        (df["first_ship_out_time"] - df["order_time"]).dt.total_seconds() / 3600.0
    )
    df["hours_to_delivery"] = (
        (df["final_arrival_time"] - df["order_time"]).dt.total_seconds() / 3600.0
    )
    return df


def build_assignment_candidates(
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
    network: pd.DataFrame,
) -> pd.DataFrame:
    order_lines = add_order_value_fields(orders).copy()
    order_lines["order_line_id"] = (
        order_lines["order_ID"].astype(str) + "_" + order_lines["sku_ID"].astype(str)
    )

    destination_region = network.rename(columns={"dc_ID": "dc_des"})
    candidate_lookup = network.rename(columns={"dc_ID": "candidate_dc"})

    candidates = order_lines.merge(
        destination_region,
        how="left",
        on="dc_des",
        validate="m:1",
    ).merge(
        candidate_lookup,
        how="left",
        on="region_ID",
        validate="m:m",
    )

    inventory_view = inventory.rename(columns={"dc_ID": "candidate_dc", "date": "order_date"})
    candidates = candidates.merge(
        inventory_view.assign(inventory_available=1),
        how="left",
        on=["candidate_dc", "sku_ID", "order_date"],
        validate="m:m",
    )

    candidates["inventory_available"] = candidates["inventory_available"].fillna(0).astype(int)
    candidates["candidate_is_current_origin"] = (
        candidates["candidate_dc"] == candidates["dc_ori"]
    ).astype(int)
    candidates["candidate_is_destination_dc"] = (
        candidates["candidate_dc"] == candidates["dc_des"]
    ).astype(int)
    candidates["candidate_remote_flag"] = (
        candidates["candidate_dc"] != candidates["dc_des"]
    ).astype(int)
    return candidates

