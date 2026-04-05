# Processed tables (`data/processed/`)

*中文说明见 [README_cn.md](README_cn.md).*

The four CSV files in this folder are produced by `notebooks/01_data_processing.ipynb` (or an equivalent pipeline) via `jd_project.features`, i.e. the output of `save_processed_tables()`. Upstream inputs are the six raw tables in `data/raw/` after cleaning through `jd_project.data.load_raw_tables`.

---

## 1. `delivery_summary.csv`

**Purpose**  
Aggregates package-level logistics into an **order-level** timeline so it can be joined to orders on `order_id`.

**Grain**  
One row per `order_id`.

**Main columns**

| Column | Description |
|--------|-------------|
| `order_id` | Order identifier |
| `package_count` | Number of distinct `package_id` values for the order |
| `first_ship_out_time` | Earliest ship-out time across packages |
| `first_arr_station_time` | Earliest arrival-at-station time across packages |
| `final_arrival_time` | Latest final delivery time across packages |

**Implementation**  
`build_delivery_summary()` in `src/jd_project/features.py`.

---

## 2. `order_line_fact.csv`

**Purpose**  
Primary fact table: each cleaned **order line** from `orders`, left-joined to users, SKUs, and the delivery summary above, with derived monetary, discount, fulfillment, and timing fields.

**Grain**  
One row per row in cleaned `orders` (multiple rows per `order_id` + `sku_id` are allowed).  
Primary key is **`order_line_id`**, formatted as `{order_id}_{sku_id}_{k}`, where `k` is a 0-based line index within the same order and SKU (row order preserved after merges).

**Column groups**

- **Order line (from `orders`)**  
  `order_id`, `user_id`, `sku_id`, `order_date`, `order_time`, `quantity`, `type`, `promise`, price and discount columns, `gift_item`, `dc_ori`, `dc_des`, `discount_total_per_unit`, `discount_gap`, `discount_consistency_flag`

- **User attributes (left join from `users`)**  
  e.g. `user_level`, `first_order_month`, `plus`, `gender`, `age`, `marital_status`, `education`, `city_level`, `purchase_power` (see actual CSV headers).

- **SKU attributes (left join from `skus`; dimension deduplicated on `sku_id`)**  
  e.g. `type_sku`, `brand_id`, `attribute1`, `attribute2`, `activate_date`, `deactivate_date`.

- **Delivery summary (left join)**  
  `package_count`, `first_ship_out_time`, `first_arr_station_time`, `final_arrival_time`

- **Derived fields**  
  `order_line_id`, `gross_merchandise_value`, `net_merchandise_value`, `discount_rate`, `is_discounted`, `remote_fulfillment_flag`, `order_hour`, `order_weekday`, `order_weekend_flag`, `promise_days`, `delivered_by_jd_flag`, `hours_to_ship`, `hours_to_delivery`

**Notes**

- `remote_fulfillment_flag`: whether `dc_ori` differs from `dc_des` (remote-shipment proxy).  
- `delivered_by_jd_flag`: whether a delivery-summary row exists (observable package-level logistics).  
- `hours_to_*`: hours from `order_time` to logistics timestamps; missing when timestamps are missing.

**Implementation**  
`build_order_line_fact()`.

---

## 3. `order_line_with_inventory.csv`

**Purpose**  
Extends `order_line_fact` with **destination / origin DC inventory** and **region-level** inventory aggregates for descriptive analysis and patterns such as “local stock available but shipped remotely.”

**Grain**  
Same as `order_line_fact`: one row per `order_line_id`.

**Columns added on top of `order_line_fact`**

| Column | Description |
|--------|-------------|
| `destination_region_id` | Region of the destination DC (`dc_des`) from the network mapping |
| `inventory_at_dc_des` | Availability flag for (`order_date`, `dc_des`, `sku_id`) in inventory (left join; missing keys filled with 0) |
| `inventory_at_dc_ori` | Same for (`order_date`, `dc_ori`, `sku_id`) |
| `num_available_dcs_in_region` | Sum of `inventory_available` over DCs in the destination region for that date and SKU (missing treated as 0 before aggregation) |
| `any_inventory_in_region` | Whether any in-region availability is recorded (sum > 0) |
| `local_available_but_remote_shipped` | 1 if destination DC shows availability but the line is still remotely fulfilled |

**Implementation**  
`build_inventory_features()`.

**Interpretation**  
`inventory_at_*` follows the raw inventory table: **no row for (`date`, `dc_id`, `sku_id`) does not mean “confirmed out of stock” in business terms**—only that there is no matching record in this dataset. Zeros are filled for counting and modeling convenience.

---

## 4. `assignment_candidates.csv`

**Purpose**  
Candidate table for warehouse-assignment optimization (e.g. PuLP): for each order line, one row per **candidate DC** in the **destination region**, with an inventory-availability flag on that candidate.

**Grain**  
One row per **`order_line_id` + `candidate_dc`** (every DC in the region gets a row per order line).

**Main columns**

| Column | Description |
|--------|-------------|
| `order_line_id` | Order-line primary key |
| `order_id`, `sku_id`, `user_id` | Traceability back to order and user |
| `order_date` | Order date (aligned to inventory `date`) |
| `dc_des`, `dc_ori` | Observed destination and origin DCs |
| `destination_region_id` | Region containing the destination DC |
| `candidate_dc` | Candidate shipping DC (within that region) |
| `candidate_in_destination_region_flag` | Whether the candidate lies in the destination region (1 under current build logic) |
| `candidate_remote_flag` | Whether the candidate DC differs from `dc_des` |
| `inventory_available` | Availability for (`order_date`, `candidate_dc`, `sku_id`); 0 when no inventory row matches |

**Implementation**  
`build_assignment_candidates()`. The network table is deduplicated on `(region_id, dc_id)` before expansion to avoid duplicate candidate rows.

**Interpretation**  
Most `inventory_available = 0` values come from **non-matches on the left join** (no row in inventory for that triple), not necessarily from an explicit “unavailable” record. State the operational definition clearly in reports or papers.

---

## Regenerating the files

After `data/raw/` is populated, run the data-processing notebook in order:

- `build_delivery_summary` → `build_order_line_fact` → `build_inventory_features` → `build_assignment_candidates`  
- then `save_processed_tables(output_dir=..., ...)`

This overwrites the four CSVs in this directory.
