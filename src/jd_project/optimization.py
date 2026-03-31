from __future__ import annotations

import pandas as pd
import pulp


def prepare_candidate_costs(
    candidates: pd.DataFrame,
    remote_penalty: float = 1.0,
    stockout_penalty: float = 1000.0,
) -> pd.DataFrame:
    df = candidates.copy()
    df["assignment_cost"] = remote_penalty * df["candidate_remote_flag"]
    df["assignment_cost"] += stockout_penalty * (1 - df["inventory_available"])
    return df


def solve_warehouse_assignment(
    candidates: pd.DataFrame,
    capacity_by_dc_day: pd.DataFrame | None = None,
    time_limit_seconds: int | None = None,
) -> tuple[pulp.LpProblem, pd.DataFrame]:
    required_cols = {
        "order_line_id",
        "candidate_dc",
        "order_date",
        "assignment_cost",
        "inventory_available",
    }
    missing = required_cols - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates missing required columns: {sorted(missing)}")

    df = candidates.copy()
    df = df[df["inventory_available"] == 1].copy()
    if df.empty:
        raise ValueError("No feasible candidate assignments remain after inventory filtering.")

    problem = pulp.LpProblem("jd_assignment", pulp.LpMinimize)

    decision_index = list(df.index)
    x = pulp.LpVariable.dicts("assign", decision_index, lowBound=0, upBound=1, cat="Binary")

    problem += pulp.lpSum(df.loc[i, "assignment_cost"] * x[i] for i in decision_index)

    for order_line_id, group in df.groupby("order_line_id").groups.items():
        group_index = list(group)
        problem += pulp.lpSum(x[i] for i in group_index) == 1, f"assign_once_{order_line_id}"

    if capacity_by_dc_day is not None:
        cap = capacity_by_dc_day.copy()
        cap["capacity_key"] = (
            cap["candidate_dc"].astype(str) + "_" + pd.to_datetime(cap["order_date"]).astype(str)
        )
        df["capacity_key"] = df["candidate_dc"].astype(str) + "_" + df["order_date"].astype(str)
        capacity_lookup = cap.set_index("capacity_key")["capacity"]

        for capacity_key, group in df.groupby("capacity_key").groups.items():
            if capacity_key not in capacity_lookup.index:
                continue
            group_index = list(group)
            problem += (
                pulp.lpSum(x[i] for i in group_index) <= float(capacity_lookup.loc[capacity_key]),
                f"capacity_{capacity_key}",
            )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    problem.solve(solver)

    solution = df[["order_line_id", "candidate_dc", "order_date", "assignment_cost"]].copy()
    solution["selected"] = [int(round(pulp.value(x[i]))) for i in decision_index]
    solution = solution[solution["selected"] == 1].reset_index(drop=True)
    return problem, solution

