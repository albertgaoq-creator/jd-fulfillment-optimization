# Processed tables (`data/processed/`)

*English: [README.md](README.md).*

本目录下的四个 CSV 由 `notebooks/01_data_processing.ipynb`（或等价流程）调用 `jd_project.features` 生成，对应 `save_processed_tables()` 的输出。上游数据来自 `data/raw/` 中经 `jd_project.data.load_raw_tables` 清洗后的六张原始表。

---

## 1. `delivery_summary.csv`

**作用**  
把包裹粒度的物流表聚合为**订单级**时间线，便于与订单表按 `order_id` 对齐。

**粒度**  
一行 = 一个 `order_id`。

**主要列**

| 列名 | 含义 |
|------|------|
| `order_id` | 订单标识 |
| `package_count` | 该订单不同 `package_id` 的个数 |
| `first_ship_out_time` | 全包裹最早出库时间 |
| `first_arr_station_time` | 全包裹最早到达站点时间 |
| `final_arrival_time` | 全包裹最晚妥投时间 |

**生成函数**  
`build_delivery_summary()`（见 `src/jd_project/features.py`）。

---

## 2. `order_line_fact.csv`

**作用**  
项目主事实表：在清洗后的**订单明细行**上合并用户、SKU、上述物流汇总，并计算金额、折扣、履约与时间类衍生字段。

**粒度**  
一行 = 清洗后 `orders` 中的一行（同一 `order_id` + `sku_id` 可有多行）。  
主键为 **`order_line_id`**，格式为 `{order_id}_{sku_id}_{k}`，`k` 为同单同 SKU 下的行序号（从 0 起）。

**列分组说明**

- **订单与商品（来自 orders）**  
  `order_id`, `user_id`, `sku_id`, `order_date`, `order_time`, `quantity`, `type`, `promise`, 价格与各类折扣列, `gift_item`, `dc_ori`, `dc_des`, `discount_total_per_unit`, `discount_gap`, `discount_consistency_flag`

- **用户属性（来自 users，左连接）**  
  如 `user_level`, `first_order_month`, `plus`, `gender`, `age`, `marital_status`, `education`, `city_level`, `purchase_power` 等（以实际 CSV 表头为准）。

- **SKU 属性（来自 skus，左连接；`sku_id` 在维表侧已去重）**  
  如 `type_sku`, `brand_id`, `attribute1`, `attribute2`, `activate_date`, `deactivate_date` 等。

- **物流汇总（来自 delivery_summary，左连接）**  
  `package_count`, `first_ship_out_time`, `first_arr_station_time`, `final_arrival_time`

- **衍生字段**  
  `order_line_id`, `gross_merchandise_value`, `net_merchandise_value`, `discount_rate`, `is_discounted`, `remote_fulfillment_flag`, `order_hour`, `order_weekday`, `order_weekend_flag`, `promise_days`, `delivered_by_jd_flag`, `hours_to_ship`, `hours_to_delivery`

**说明**

- `remote_fulfillment_flag`：`dc_ori` 与 `dc_des` 是否不同（异地发货代理）。  
- `delivered_by_jd_flag`：是否存在对应物流汇总（有包裹级记录则视为京东配送侧可观测）。  
- `hours_to_*`：由 `order_time` 与物流时间戳计算的小时数，缺失时间为空。

**生成函数**  
`build_order_line_fact()`。

---

## 3. `order_line_with_inventory.csv`

**作用**  
在 `order_line_fact` 基础上增加**目的仓 / 始发仓库存**以及**目的区域**内的库存聚合指标，用于描述性分析与「本可本地发却异地发」等模式。

**粒度**  
与 `order_line_fact` 相同：一行 = 一个 `order_line_id`。

**在 `order_line_fact` 基础上新增的列**

| 列名 | 含义 |
|------|------|
| `destination_region_id` | 由 `dc_des` 经 network 映射得到的目的区域 |
| `inventory_at_dc_des` | 当日、目的仓 `dc_des`、该 `sku_id` 在库存表中的可用标记（左连接，无记录填 0） |
| `inventory_at_dc_ori` | 当日、始发仓 `dc_ori`、该 `sku_id` 的可用标记（同上） |
| `num_available_dcs_in_region` | 目的区域内各 DC 在该日该 SKU 上的 `inventory_available` 之和（实现上缺失先按 0 再聚合） |
| `any_inventory_in_region` | 区域内是否存在可用库存记录（计数 > 0） |
| `local_available_but_remote_shipped` | 目的仓有货且仍为异地发货时为 1 |

**生成函数**  
`build_inventory_features()`。

**解读注意**  
`inventory_at_*` 与原始库存表一致：无 `(date, dc_id, sku_id)` 行时**不表示业务上确认无货**，仅表示本数据集中无匹配记录；代码中用 0 填充便于计数与建模。

---

## 4. `assignment_candidates.csv`

**作用**  
为仓配分配类优化（如 PuLP）准备**候选方案表**：每条订单行在**目的区域**内对每个候选配送中心（DC）占一行，并挂上该候选上的库存可用标记。

**粒度**  
一行 = **`order_line_id` + `candidate_dc`**（同一订单行在区域内每个 DC 各一行）。

**主要列**

| 列名 | 含义 |
|------|------|
| `order_line_id` | 订单行主键 |
| `order_id`, `sku_id`, `user_id` | 便于回溯订单与用户 |
| `order_date` | 订单日（与库存 `date` 对齐） |
| `dc_des`, `dc_ori` | 实际目的仓、始发仓 |
| `destination_region_id` | 目的仓所在区域 |
| `candidate_dc` | 候选发货仓（该区域内的 DC） |
| `candidate_in_destination_region_flag` | 候选是否属于目的区域（当前构建逻辑下为 1） |
| `candidate_remote_flag` | 候选仓是否与 `dc_des` 不同 |
| `inventory_available` | 该日、`candidate_dc`、`sku_id` 在库存表中的可用标记；无匹配记录为 0 |

**生成函数**  
`build_assignment_candidates()`。Network 表在展开前对 `(region_id, dc_id)` 去重，避免重复候选行。

**解读注意**  
`inventory_available = 0` 多数来自**左连接未命中**（该组合在库存表中无行），与「明确记录为无货」需区分；报告或论文中建议写明操作化定义。

---

## 重新生成

在配置好 `data/raw/` 后运行数据处理笔记本中调用：

- `build_delivery_summary` → `build_order_line_fact` → `build_inventory_features` → `build_assignment_candidates`  
- 最后 `save_processed_tables(output_dir=..., ...)`

即可覆盖更新本目录下四个 CSV。
