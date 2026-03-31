from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path(os.getenv("JD_RAW_DATA_DIR", DATA_DIR / "raw"))
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

JD_FILENAMES = {
    "orders": "JD_order_data.csv",
    "delivery": "JD_delivery_data.csv",
    "inventory": "JD_inventory_data.csv",
    "network": "JD_network_data.csv",
    "users": "JD_user_data.csv",
    "skus": "JD_sku_data.csv",
    "clicks": "JD_click_data.csv",
}


def ensure_project_dirs() -> None:
    for path in [DATA_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)

