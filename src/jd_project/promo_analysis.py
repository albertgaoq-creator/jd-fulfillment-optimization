import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_plot(title, save_dir):
    filename = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    if save_dir is not None:
        plt.savefig(save_dir / f"{filename}.png", dpi=300)


def run_promo_analysis(orders_df, show_plots=True, verbose=True, save_dir=None):
    """
    Run promotion effectiveness analysis on the JD order table.

    Parameters
    ----------
    orders_df : pandas.DataFrame
        The JD orders table.
    show_plots : bool, default=True
        Whether to display plots.
    verbose : bool, default=True
        Whether to print intermediate summaries.
    save_dir : pathlib.Path or None, default=None
        Directory to save plot PNG files. If None, plots are shown but not saved.

    Returns
    -------
    dict
        A dictionary containing:
        - promo_df
        - reconcile_summary
        - discount_bin_summary
        - discount_frequency
        - discount_amount
        - discount_overview
        - seller_summary
        - usage_by_type_df
        - amount_by_type_df
        - usage_pivot
        - amount_pivot
        - summary_table
        - promo_only_summary
        - correlations
        - reconciliation_bad_ratio
    """

    # ------------------------------------------------------------
    # 1. Copy the input dataframe
    # ------------------------------------------------------------
    promo_df = orders_df.copy()

    # ------------------------------------------------------------
    # 2. Define discount-related columns
    # ------------------------------------------------------------
    discount_cols = [
        "direct_discount_per_unit",
        "quantity_discount_per_unit",
        "bundle_discount_per_unit",
        "coupon_discount_per_unit",
    ]

    required_cols = discount_cols + [
        "original_unit_price",
        "final_unit_price",
        "quantity",
        "type",
        "order_id",
    ]

    missing_cols = [col for col in required_cols if col not in promo_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in orders_df: {missing_cols}")

    # ------------------------------------------------------------
    # 3. Convert relevant columns to numeric
    # ------------------------------------------------------------
    numeric_cols = discount_cols + [
        "original_unit_price",
        "final_unit_price",
        "quantity",
        "type",
    ]

    for col in numeric_cols:
        promo_df[col] = pd.to_numeric(promo_df[col], errors="coerce")

    promo_df[discount_cols] = promo_df[discount_cols].fillna(0)
    promo_df["quantity"] = promo_df["quantity"].fillna(0)
    promo_df["original_unit_price"] = promo_df["original_unit_price"].fillna(0)
    promo_df["final_unit_price"] = promo_df["final_unit_price"].fillna(0)

    # ------------------------------------------------------------
    # 4. Build promotion metrics
    # ------------------------------------------------------------
    promo_df["gross_merch_value"] = promo_df["original_unit_price"] * promo_df["quantity"]
    promo_df["net_merch_value"] = promo_df["final_unit_price"] * promo_df["quantity"]

    promo_df["total_discount_per_unit"] = promo_df[discount_cols].sum(axis=1)
    promo_df["total_discount_amount"] = promo_df["total_discount_per_unit"] * promo_df["quantity"]

    promo_df["price_gap_per_unit"] = promo_df["original_unit_price"] - promo_df["final_unit_price"]
    promo_df["price_gap_amount"] = promo_df["price_gap_per_unit"] * promo_df["quantity"]

    promo_df["discount_rate"] = np.where(
        promo_df["original_unit_price"] > 0,
        promo_df["total_discount_per_unit"] / promo_df["original_unit_price"],
        np.nan,
    )
    promo_df["discount_rate"] = promo_df["discount_rate"].clip(lower=0, upper=1)

    promo_df["seller_type"] = promo_df["type"].map({1: "1P", 2: "3P"}).fillna("Unknown")

    for col in discount_cols:
        promo_df[f"has_{col}"] = promo_df[col] > 0

    if verbose:
        print("========== Promo DataFrame ==========")
        print("Shape:", promo_df.shape)
        print()

    # ------------------------------------------------------------
    # 5. Reconciliation check
    # ------------------------------------------------------------
    promo_df["discount_reconcile_diff"] = (
        promo_df["price_gap_per_unit"] - promo_df["total_discount_per_unit"]
    )

    reconcile_summary = promo_df["discount_reconcile_diff"].describe()
    tol = 1e-6
    bad_ratio = (promo_df["discount_reconcile_diff"].abs() > tol).mean()

    if verbose:
        print("========== Reconciliation Check ==========")
        print(reconcile_summary)
        print(f"Share of rows with reconciliation mismatch > {tol}: {bad_ratio:.4%}")
        print()

    # ------------------------------------------------------------
    # 6. Discount rate vs quantity
    # ------------------------------------------------------------
    corr_df = promo_df[["discount_rate", "quantity"]].dropna()

    pearson_corr = corr_df["discount_rate"].corr(corr_df["quantity"], method="pearson")
    spearman_corr = corr_df["discount_rate"].corr(corr_df["quantity"], method="spearman")

    correlations = {
        "pearson_discount_rate_quantity": pearson_corr,
        "spearman_discount_rate_quantity": spearman_corr,
    }

    if verbose:
        print("========== Discount Rate vs Quantity ==========")
        print(f"Pearson correlation : {pearson_corr:.4f}")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print()

    promo_df["discount_rate_bin"] = pd.cut(
        promo_df["discount_rate"],
        bins=[-0.001, 0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00],
        labels=["0", "(0,5%]", "(5,10%]", "(10,20%]", "(20,30%]", "(30,50%]", ">50%"],
    )

    discount_bin_summary = (
        promo_df.groupby("discount_rate_bin", observed=False)
        .agg(
            order_lines=("order_id", "count"),
            total_units=("quantity", "sum"),
            avg_units_per_line=("quantity", "mean"),
            median_units_per_line=("quantity", "median"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        .reset_index()
    )

    if verbose:
        print("========== Discount Bucket Summary ==========")
        print(discount_bin_summary)
        print()

    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.bar(
            discount_bin_summary["discount_rate_bin"].astype(str),
            discount_bin_summary["avg_units_per_line"],
        )
        plt.xticks(rotation=30)
        plt.ylabel("Average quantity per order line")
        plt.xlabel("Discount rate bucket")
        title = "Average quantity by discount rate bucket"
        plt.title(title)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.bar(
            discount_bin_summary["discount_rate_bin"].astype(str),
            discount_bin_summary["total_units"],
        )
        plt.xticks(rotation=30)
        plt.ylabel("Total units sold")
        plt.xlabel("Discount rate bucket")
        title = "Total units by discount rate bucket"
        plt.title(title)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

    # ------------------------------------------------------------
    # 7. Discount frequency
    # ------------------------------------------------------------
    frequency_rows = []
    n_lines = len(promo_df)

    for col in discount_cols:
        used_lines = (promo_df[col] > 0).sum()
        used_ratio = used_lines / n_lines
        frequency_rows.append(
            {
                "discount_type": col,
                "used_order_lines": used_lines,
                "used_ratio": used_ratio,
            }
        )

    discount_frequency = pd.DataFrame(frequency_rows).sort_values(
        "used_order_lines", ascending=False
    )

    if verbose:
        print("========== Discount Frequency ==========")
        print(discount_frequency)
        print()

    # ------------------------------------------------------------
    # 8. Discount amount contribution
    # ------------------------------------------------------------
    amount_rows = []

    for col in discount_cols:
        total_amount = (promo_df[col] * promo_df["quantity"]).sum()

        if (promo_df[col] > 0).any():
            avg_amount_per_used_line = (
                promo_df.loc[promo_df[col] > 0, col]
                * promo_df.loc[promo_df[col] > 0, "quantity"]
            ).mean()
        else:
            avg_amount_per_used_line = 0

        amount_rows.append(
            {
                "discount_type": col,
                "total_discount_amount": total_amount,
                "avg_discount_amount_per_used_line": avg_amount_per_used_line,
            }
        )

    discount_amount = pd.DataFrame(amount_rows).sort_values(
        "total_discount_amount", ascending=False
    )

    if verbose:
        print("========== Discount Amount Contribution ==========")
        print(discount_amount)
        print()

    discount_overview = discount_frequency.merge(
        discount_amount, on="discount_type", how="left"
    )

    if verbose:
        print("========== Discount Overview ==========")
        print(discount_overview)
        print()

    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.bar(discount_overview["discount_type"], discount_overview["used_ratio"])
        plt.xticks(rotation=30)
        plt.ylabel("Usage ratio")
        plt.xlabel("Discount type")
        title = "How often each discount type appears"
        plt.title(title)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.bar(
            discount_overview["discount_type"],
            discount_overview["total_discount_amount"],
        )
        plt.xticks(rotation=30)
        plt.ylabel("Total discount amount")
        plt.xlabel("Discount type")
        title = "Total discount amount by discount type"
        plt.title(title)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

    # ------------------------------------------------------------
    # 9. 1P vs 3P comparison
    # ------------------------------------------------------------
    seller_summary = (
        promo_df.groupby("seller_type")
        .agg(
            order_lines=("order_id", "count"),
            total_units=("quantity", "sum"),
            avg_units_per_line=("quantity", "mean"),
            avg_original_price=("original_unit_price", "mean"),
            avg_final_price=("final_unit_price", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            median_discount_rate=("discount_rate", "median"),
            promo_line_ratio=("total_discount_per_unit", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    if verbose:
        print("========== Seller Type Summary ==========")
        print(seller_summary)
        print()

    usage_by_type = []

    for seller in ["1P", "3P", "Unknown"]:
        tmp = promo_df[promo_df["seller_type"] == seller]
        if len(tmp) == 0:
            continue

        for col in discount_cols:
            usage_by_type.append(
                {
                    "seller_type": seller,
                    "discount_type": col,
                    "usage_ratio": (tmp[col] > 0).mean(),
                    "used_order_lines": (tmp[col] > 0).sum(),
                }
            )

    usage_by_type_df = pd.DataFrame(usage_by_type)

    if verbose:
        print("========== Discount Usage by Seller Type ==========")
        print(usage_by_type_df)
        print()

    amount_by_type = []

    for seller in ["1P", "3P", "Unknown"]:
        tmp = promo_df[promo_df["seller_type"] == seller]
        if len(tmp) == 0:
            continue

        for col in discount_cols:
            amount_by_type.append(
                {
                    "seller_type": seller,
                    "discount_type": col,
                    "total_discount_amount": (tmp[col] * tmp["quantity"]).sum(),
                }
            )

    amount_by_type_df = pd.DataFrame(amount_by_type)

    if verbose:
        print("========== Discount Amount by Seller Type ==========")
        print(amount_by_type_df)
        print()

    usage_pivot = usage_by_type_df.pivot(
        index="discount_type", columns="seller_type", values="usage_ratio"
    )
    amount_pivot = amount_by_type_df.pivot(
        index="discount_type", columns="seller_type", values="total_discount_amount"
    )

    if verbose:
        print("========== Usage Ratio Pivot ==========")
        print(usage_pivot)
        print()

        print("========== Total Discount Amount Pivot ==========")
        print(amount_pivot)
        print()

    if show_plots:
        plot_df = usage_by_type_df[usage_by_type_df["seller_type"].isin(["1P", "3P"])]
        pivot_plot = plot_df.pivot(
            index="discount_type", columns="seller_type", values="usage_ratio"
        )
        pivot_plot.plot(kind="bar", figsize=(9, 5))
        plt.ylabel("Usage ratio")
        plt.xlabel("Discount type")
        title = "Discount usage ratio 1P vs 3P"
        plt.title(title)
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

        plot_amt_df = amount_by_type_df[
            amount_by_type_df["seller_type"].isin(["1P", "3P"])
        ]
        pivot_amt_plot = plot_amt_df.pivot(
            index="discount_type", columns="seller_type", values="total_discount_amount"
        )
        pivot_amt_plot.plot(kind="bar", figsize=(9, 5))
        plt.ylabel("Total discount amount")
        plt.xlabel("Discount type")
        title = "Discount amount contribution 1P vs 3P"
        plt.title(title)
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

    # ------------------------------------------------------------
    # 10. Final summary table
    # ------------------------------------------------------------
    summary_table = discount_overview.copy()
    summary_table["used_ratio"] = summary_table["used_ratio"].round(4)
    summary_table["total_discount_amount"] = summary_table["total_discount_amount"].round(2)
    summary_table["avg_discount_amount_per_used_line"] = (
        summary_table["avg_discount_amount_per_used_line"].round(2)
    )

    if verbose:
        print("========== Final Summary Table ==========")
        print(summary_table)
        print()

    # ------------------------------------------------------------
    # 11. Promoted lines only
    # ------------------------------------------------------------
    promo_only_df = promo_df[promo_df["total_discount_per_unit"] > 0].copy()

    promo_only_summary = (
        promo_only_df.groupby("discount_rate_bin", observed=False)
        .agg(
            order_lines=("order_id", "count"),
            total_units=("quantity", "sum"),
            avg_units_per_line=("quantity", "mean"),
            median_units_per_line=("quantity", "median"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        .reset_index()
    )

    if verbose:
        print("========== Promoted Lines Only: Discount Bucket Summary ==========")
        print(promo_only_summary)
        print()

    if show_plots:
        plot_data = promo_only_summary.copy()
        plot_data["avg_units_per_line"] = pd.to_numeric(
            plot_data["avg_units_per_line"], errors="coerce"
        ).fillna(0)

        plt.figure(figsize=(10, 5))
        plt.bar(
            plot_data["discount_rate_bin"].astype(str),
            plot_data["avg_units_per_line"],
        )
        plt.xticks(rotation=30)
        plt.ylabel("Average quantity per promoted order line")
        plt.xlabel("Discount rate bucket")
        title = "Average quantity by discount rate bucket promoted lines only"
        plt.title(title)
        plt.tight_layout()
        save_plot(title, save_dir)
        plt.show()

    return {
        "promo_df": promo_df,
        "reconcile_summary": reconcile_summary,
        "discount_bin_summary": discount_bin_summary,
        "discount_frequency": discount_frequency,
        "discount_amount": discount_amount,
        "discount_overview": discount_overview,
        "seller_summary": seller_summary,
        "usage_by_type_df": usage_by_type_df,
        "amount_by_type_df": amount_by_type_df,
        "usage_pivot": usage_pivot,
        "amount_pivot": amount_pivot,
        "summary_table": summary_table,
        "promo_only_summary": promo_only_summary,
        "correlations": correlations,
        "reconciliation_bad_ratio": bad_ratio,
    }