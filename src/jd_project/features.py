"""Feature-building helpers for Part A of the JD final project.

This module assumes the input tables have already been cleaned by
`src/jd_project/data.py`. The functions here build processed tables that are
ready for downstream analysis in notebooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_required_columns(df: pd.DataFrame, required_columns: list[str], table_name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{table_name} is missing required columns: {missing_text}")


def _safe_hours_between(end_series: pd.Series, start_series: pd.Series) -> pd.Series:
    """Return elapsed hours between two datetime series, or NaN when unavailable."""
    return (end_series - start_series).dt.total_seconds() / 3600.0


def _build_destination_region_lookup(network_df: pd.DataFrame) -> pd.DataFrame:
    """Map each destination DC to its region ID using the cleaned network table."""
    _find_required_columns(network_df, ["region_id", "dc_id"], "network_df")
    lookup = network_df[["dc_id", "region_id"]].drop_duplicates().rename(
        columns={"dc_id": "dc_des", "region_id": "destination_region_id"}
    )
    return lookup


def _network_region_dc_unique(network_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (region_id, dc_id) to avoid duplicate candidate rows from repeated edges."""
    _find_required_columns(network_df, ["region_id", "dc_id"], "network_df")
    return network_df[["region_id", "dc_id"]].drop_duplicates().reset_index(drop=True)


def build_delivery_summary(delivery_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate package-level delivery data to one row per order.

    Notes
    -----
    This is an ORDER-LEVEL proxy built from package-level delivery records.
    If an order has multiple packages, this summary keeps the first ship/station
    timestamps and the final arrival timestamp across all packages.
    """
    _find_required_columns(
        delivery_df,
        ["order_id", "package_id", "ship_out_time", "arr_station_time", "arr_time"],
        "delivery_df",
    )

    summary = (
        delivery_df.groupby("order_id", as_index=False)
        .agg(
            package_count=("package_id", "nunique"),
            first_ship_out_time=("ship_out_time", "min"),
            first_arr_station_time=("arr_station_time", "min"),
            final_arrival_time=("arr_time", "max"),
        )
        .sort_values("order_id")
        .reset_index(drop=True)
    )
    return summary


def build_order_line_fact(
    orders_df: pd.DataFrame,
    users_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    delivery_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main order-line fact table (one row per cleaned orders row).

    ``order_line_id`` is unique: ``{order_id}_{sku_id}_{k}`` where ``k`` is the
    0-based index among rows with the same ``order_id`` and ``sku_id`` (order
    follows the current dataframe row order after merges).
    """
    _find_required_columns(
        orders_df,
        [
            "order_id",
            "user_id",
            "sku_id",
            "order_date",
            "order_time",
            "quantity",
            "original_unit_price",
            "final_unit_price",
            "discount_total_per_unit",
            "dc_ori",
            "dc_des",
        ],
        "orders_df",
    )
    _find_required_columns(users_df, ["user_id"], "users_df")
    _find_required_columns(skus_df, ["sku_id"], "skus_df")
    _find_required_columns(
        delivery_summary_df,
        [
            "order_id",
            "package_count",
            "first_ship_out_time",
            "first_arr_station_time",
            "final_arrival_time",
        ],
        "delivery_summary_df",
    )

    out = orders_df.copy()
    out = out.merge(users_df, how="left", on="user_id", validate="m:1")

    # SKU extract can repeat sku_id; merge requires exactly one row per key on the right.
    # Normalize keys so values that only differ by whitespace/type do not stay as false duplicates.
    skus_right = skus_df.copy()
    skus_right["sku_id"] = skus_right["sku_id"].astype("string").str.strip()
    skus_unique = skus_right.drop_duplicates(subset=["sku_id"], keep="first").reset_index(drop=True)
    out["sku_id"] = out["sku_id"].astype("string").str.strip()
    out = out.merge(skus_unique, how="left", on="sku_id", validate="m:1", suffixes=("", "_sku"))
    out = out.merge(delivery_summary_df, how="left", on="order_id", validate="m:1")

    oid = out["order_id"].astype("string").str.strip()
    sid = out["sku_id"]
    line_ix = (
        pd.DataFrame({"_oid": oid, "_sid": sid})
        .groupby(["_oid", "_sid"], sort=False)
        .cumcount()
    )
    out["order_line_id"] = oid + "_" + sid + "_" + line_ix.astype("string")

    out["gross_merchandise_value"] = out["quantity"] * out["original_unit_price"]
    out["net_merchandise_value"] = out["quantity"] * out["final_unit_price"]
    out["discount_rate"] = out["discount_total_per_unit"] / out["original_unit_price"].replace(0, pd.NA)
    out["is_discounted"] = (out["discount_total_per_unit"].fillna(0) > 0).astype("Int64")
    out["remote_fulfillment_flag"] = (out["dc_ori"] != out["dc_des"]).astype("Int64")

    out["order_hour"] = out["order_time"].dt.hour
    out["order_weekday"] = out["order_time"].dt.dayofweek
    out["order_weekend_flag"] = out["order_weekday"].isin([5, 6]).astype("Int64")

    if "promise" in out.columns:
        out["promise_days"] = out["promise"]
    else:
        out["promise_days"] = pd.NA

    # Proxy: an order with package-level delivery records is treated as delivered by JD.
    out["delivered_by_jd_flag"] = out["package_count"].notna().astype("Int64")

    # Leave NaN if either timestamp is missing or invalid.
    out["hours_to_ship"] = _safe_hours_between(out["first_ship_out_time"], out["order_time"])
    out["hours_to_delivery"] = _safe_hours_between(out["final_arrival_time"], out["order_time"])

    return out


def build_inventory_features(
    order_line_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    network_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add inventory- and region-based fulfillment features to the order-line fact table."""
    _find_required_columns(
        order_line_df,
        ["order_line_id", "order_date", "sku_id", "dc_des", "dc_ori", "remote_fulfillment_flag"],
        "order_line_df",
    )
    _find_required_columns(inventory_df, ["date", "dc_id", "sku_id", "inventory_available"], "inventory_df")
    _find_required_columns(network_df, ["region_id", "dc_id"], "network_df")

    out = order_line_df.copy()

    destination_lookup = _build_destination_region_lookup(network_df)
    out = out.merge(destination_lookup, how="left", on="dc_des", validate="m:1")

    inventory_lookup = inventory_df[["date", "dc_id", "sku_id", "inventory_available"]].drop_duplicates()

    inventory_at_des = inventory_lookup.rename(
        columns={
            "date": "order_date",
            "dc_id": "dc_des",
            "inventory_available": "inventory_at_dc_des",
        }
    )
    out = out.merge(
        inventory_at_des,
        how="left",
        on=["order_date", "dc_des", "sku_id"],
        validate="m:1",
    )

    inventory_at_ori = inventory_lookup.rename(
        columns={
            "date": "order_date",
            "dc_id": "dc_ori",
            "inventory_available": "inventory_at_dc_ori",
        }
    )
    out = out.merge(
        inventory_at_ori,
        how="left",
        on=["order_date", "dc_ori", "sku_id"],
        validate="m:1",
    )

    network_unique = _network_region_dc_unique(network_df)
    region_inventory = network_unique.rename(columns={"dc_id": "candidate_dc"}).merge(
        inventory_lookup.rename(columns={"dc_id": "candidate_dc", "date": "order_date"}),
        how="left",
        on=["candidate_dc"],
        validate="1:m",
    )
    region_inventory["inventory_available"] = region_inventory["inventory_available"].fillna(0)

    region_counts = (
        region_inventory.groupby(["region_id", "order_date", "sku_id"], as_index=False)
        .agg(num_available_dcs_in_region=("inventory_available", "sum"))
        .rename(columns={"region_id": "destination_region_id"})
    )

    out = out.merge(
        region_counts,
        how="left",
        on=["destination_region_id", "order_date", "sku_id"],
        validate="m:1",
    )

    out["inventory_at_dc_des"] = out["inventory_at_dc_des"].fillna(0).astype("Int64")
    out["inventory_at_dc_ori"] = out["inventory_at_dc_ori"].fillna(0).astype("Int64")
    out["num_available_dcs_in_region"] = out["num_available_dcs_in_region"].fillna(0).astype("Int64")
    out["any_inventory_in_region"] = (out["num_available_dcs_in_region"] > 0).astype("Int64")
    out["local_available_but_remote_shipped"] = (
        (out["inventory_at_dc_des"] == 1) & (out["remote_fulfillment_flag"] == 1)
    ).astype("Int64")

    return out


def build_assignment_candidates(
    order_line_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    network_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate warehouse assignments at grain `order_line_id + candidate_dc`."""
    _find_required_columns(
        order_line_df,
        ["order_line_id", "order_id", "sku_id", "order_date", "dc_des", "dc_ori"],
        "order_line_df",
    )
    _find_required_columns(inventory_df, ["date", "dc_id", "sku_id", "inventory_available"], "inventory_df")
    _find_required_columns(network_df, ["region_id", "dc_id"], "network_df")

    base_columns = ["order_line_id", "order_id", "sku_id", "order_date", "dc_des", "dc_ori"]
    if "user_id" in order_line_df.columns:
        base_columns.append("user_id")

    out = order_line_df[base_columns].copy()
    destination_lookup = _build_destination_region_lookup(network_df)
    out = out.merge(destination_lookup, how="left", on="dc_des", validate="m:1")

    candidates = _network_region_dc_unique(network_df).rename(
        columns={"dc_id": "candidate_dc", "region_id": "destination_region_id"}
    )
    out = out.merge(candidates, how="left", on="destination_region_id", validate="m:m")

    inventory_lookup = inventory_df.rename(
        columns={"date": "order_date", "dc_id": "candidate_dc"}
    )[["order_date", "candidate_dc", "sku_id", "inventory_available"]].drop_duplicates()

    out = out.merge(
        inventory_lookup,
        how="left",
        on=["order_date", "candidate_dc", "sku_id"],
        validate="m:1",
    )

    out["candidate_in_destination_region_flag"] = 1
    out["candidate_remote_flag"] = (out["candidate_dc"] != out["dc_des"]).astype("Int64")
    out["inventory_available"] = out["inventory_available"].fillna(0).astype("Int64")

    ordered_columns = [
        "order_line_id",
        "order_id",
        "sku_id",
        "order_date",
        "dc_des",
        "dc_ori",
        "destination_region_id",
        "candidate_dc",
        "candidate_in_destination_region_flag",
        "candidate_remote_flag",
        "inventory_available",
    ]
    if "user_id" in out.columns:
        ordered_columns.insert(3, "user_id")

    out = out[ordered_columns].sort_values(["order_line_id", "candidate_dc"]).reset_index(drop=True)
    return out


def save_processed_tables(
    output_dir: str | Path,
    delivery_summary_df: pd.DataFrame,
    order_line_fact_df: pd.DataFrame,
    order_line_with_inventory_df: pd.DataFrame,
    assignment_candidates_df: pd.DataFrame,
) -> None:
    """Save the four processed tables to CSV files in the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    delivery_summary_df.to_csv(output_path / "delivery_summary.csv", index=False)
    order_line_fact_df.to_csv(output_path / "order_line_fact.csv", index=False)
    order_line_with_inventory_df.to_csv(output_path / "order_line_with_inventory.csv", index=False)
    assignment_candidates_df.to_csv(output_path / "assignment_candidates.csv", index=False)


def run_basic_quality_checks(
    delivery_summary_df: pd.DataFrame,
    order_line_fact_df: pd.DataFrame,
    order_line_with_inventory_df: pd.DataFrame,
    assignment_candidates_df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Return a few lightweight row-count and duplication checks."""
    return {
        "delivery_summary": {
            "row_count": int(len(delivery_summary_df)),
            "duplicate_order_id": int(delivery_summary_df.duplicated(subset=["order_id"]).sum()),
        },
        "order_line_fact": {
            "row_count": int(len(order_line_fact_df)),
            "duplicate_order_line_id": int(order_line_fact_df.duplicated(subset=["order_line_id"]).sum()),
        },
        "order_line_with_inventory": {
            "row_count": int(len(order_line_with_inventory_df)),
            "duplicate_order_line_id": int(
                order_line_with_inventory_df.duplicated(subset=["order_line_id"]).sum()
            ),
        },
        "assignment_candidates": {
            "row_count": int(len(assignment_candidates_df)),
            "duplicate_order_line_candidate": int(
                assignment_candidates_df.duplicated(subset=["order_line_id", "candidate_dc"]).sum()
            ),
        },
    }

