
## Business Use Case
Synthetic operational warehouse utilization dataset for King Abdulaziz International Airport (JED), supporting:
- Warehouse occupancy forecasting
- Congestion prediction
- Dynamic storage reallocation
- Cold chain and DG monitoring
- Export build-up optimization

## Dataset Characteristics
- Coverage: 2025-03-01 to 2026-02-28
- Frequency: Hourly
- Total Records: 43,800
- Features: 21

## Operational Footprints
1. General Import
2. General Export
3. Cold Chain Import
4. Dangerous Goods Import
5. VIP Export

## Core Equations

available_space_m3 = zone_capacity_m3 - occupied_volume_m3

occupancy_rate = (occupied_volume_m3 / zone_capacity_m3) * 100

## Capacity Mapping

| Storage Type | Flow Direction | Capacity |
|---|---|---|
| General | Import | 5500 |
| General | Export | 4500 |
| Cold Chain | Any | 1500 |
| Dangerous Goods | Any | 1000 |
| VIP | Any | 600 |

## Route and SHC Mapping

| Route Codes | SHC | Storage Type |
|---|---|---|
| AMS / NBO | PER | Cold Chain |
| FRA / CDG | PIL | Cold Chain |
| PVG / HKG | VAL | VIP |
| DXB / DOH | DGR | Dangerous Goods |
| Others | GEN | General |

## Verification Snippet

```python
import pandas as pd

df = pd.read_csv("jeddah_air_cargo_occupancy_master.csv")

assert (
    df["available_space_m3"].round(2) ==
    (df["zone_capacity_m3"] - df["occupied_volume_m3"]).round(2)
).all()

assert (
    df["occupancy_rate"].round(4) ==
    ((df["occupied_volume_m3"] / df["zone_capacity_m3"]) * 100).round(4)
).all()

assert df.isnull().sum().sum() == 0
```

## Notes
- Q4 seasonal uplift multiplier applied during November and December.
- Weekend bottleneck effects applied Friday 12:00 through Saturday 23:59.
- Freighter arrival waves increase congestion and ULD density.
- Export build-up logic activates near departure windows.


# Technical Documentation & Functional Data Dictionary
## Use Case WH-001: Warehouse Space Utilization Predictive Engine
### King Abdulaziz International Airport (JED) – Jeddah Air Cargo Terminal
### Core System Blueprint: Two-Feature Geographic Refactor

---

## 1. System Scale, Boundaries & Dataset Topology
To evaluate and simulate terminal storage physics accurately, the data pipeline establishes strict boundaries across a continuous chronological timeline. 

* **Temporal Horizon:** Exactly one full operational year spanning from **March 1, 2025, at 00:00:00 to February 28, 2026, at 23:00:00**.
* **Temporal Granularity:** Strict hourly intervals ($\Delta t = 1\text{H}$).
* **Spatial Footprint:** 5 distinct, isolated warehouse terminal locations represented dynamically by the intersection of `flow_direction` and `storage_type`.
* **Dataset Shape:** Exactly **43,800 rows** ($8,760 \text{ hours} \times 5 \text{ active storage areas}$). The file must contain a continuous sequence with zero missing index values, zero empty hours, and zero row duplications.

---

## 2. Structural Mass-Balance Identity Equations
To preserve physical realism, every record must enforce absolute mathematical consistency. The relationships between current cargo density, physical volume, and remaining open space are governed by deterministic equations that are evaluated on every row:

### The Space Availability Conservation Identity:
$$\text{available\_space\_m3} = \text{zone\_capacity\_m3} - \text{occupied\_volume\_m3}$$

### The System Core Target Identity:
$$\text{occupancy\_rate} = \left( \frac{\text{occupied\_volume\_m3}}{\text{zone\_capacity\_m3}} \right) \times 100$$

### Static Capacity Hardware Allocations (`zone_capacity_m3` Caps):
Each room type represents a distinct physical infrastructure footprint with fixed capacity limits:
1. **General Storage Area (`storage_type` == "General"):**
   * **Inbound Terminal (`flow_direction` == "Import"):** $\text{zone\_capacity\_m3} = 5,500\text{ m}^3$ (High-turnover footprint for local container breakdowns).
   * **Outbound Staging Area (`flow_direction` == "Export"):** $\text{zone\_capacity\_m3} = 4,500\text{ m}^3$ (Heavy volume area dedicated to flight container assembly).
2. **Cold Chain Storage Unit (`storage_type` == "Cold Chain"):** $\text{zone\_capacity\_m3} = 1,500\text{ m}^3$ (Temperature-regulated climate cells).
3. **Dangerous Goods Vault (`storage_type` == "Dangerous Goods"):** $\text{zone\_capacity\_m3} = 1,000\text{ m}^3$ (Reinforced, secure hazardous storage cells).
4. **VIP Security Vault (`storage_type` == "VIP"):** $\text{zone\_capacity\_m3} = 600\text{ m}^3$ (High-value micro-storage zone).

---

## 3. High-Fidelity Categorical Combinations & Mapping Matrix
The features within this data matrix are structurally tied to one another. Selecting a specific transit route or workflow direction enforces a cascade of deterministic mappings across special handling codes, temperature settings, and process flags.



### Complete Flight Route & Special Handling Code (SHC) Conditional Rules:
* **`AMS-JED` (Amsterdam to Jeddah) & `JED-AMS`:** Formatted by direction. Bounded to a $75\%$ probability of cargo matching IATA SHC = **`PER`** (Perishables). This combination forces `storage_type` = **`"Cold Chain"`** and `temp_range` = **`"2-8°C"`**. The remaining $25\%$ maps to standard general freight.
* **`NBO-JED` (Nairobi to Jeddah) & `JED-NBO`:** Bounded to a $75\%$ probability of cargo matching IATA SHC = **`PER`** (Fresh Produce). This combination forces `storage_type` = **`"Cold Chain"`** and `temp_range` = **`"2-8°C"`**.
* **`FRA-JED` (Frankfurt to Jeddah) & `JED-FRA`:** Bounded to a $60\%$ probability of matching IATA SHC = **`PIL`** (Temperature-Controlled Pharmaceuticals). This combination forces `storage_type` = **`"Cold Chain"`** and `temp_range` = **`"15-25°C"`**.
* **`CDG-JED` (Paris to Jeddah) & `JED-CDG`:** Bounded to a $60\%$ probability of matching IATA SHC = **`PIL`** (Medical Vaccines/Pharma). This combination forces `storage_type` = **`"Cold Chain"`** and `temp_range` = **`"15-25°C"`**.
* **`PVG-JED` (Shanghai to Jeddah) & `JED-PVG`:** Bounded to a $70\%$ probability of matching IATA SHC = **`VAL`** (High-Value Electronics/ Bullion). This combination forces `storage_type` = **`"VIP"`** and `temp_range` = **`"Ambient"`**.
* **`HKG-JED` (Hong Kong to Jeddah) & `JED-HKG`:** Bounded to a $70\%$ probability of matching IATA SHC = **`VAL`** (Precious Metals/Premium Goods). This combination forces `storage_type` = **`"VIP"`** and `temp_range` = **`"Ambient"`**.
* **`DXB-JED` (Dubai to Jeddah) & `JED-DXB`:** Bounded to a $25\%$ probability of matching IATA SHC = **`DGR`** (Dangerous Goods Classes 1–9). This combination forces `storage_type` = **`"Dangerous Goods"`** and `temp_range` = **`"Ambient"`**. The remaining $75\%$ defaults to standard commercial freight.
* **`DOH-JED` (Doha to Jeddah) & `JED-DOH`:** Bounded to a $25\%$ probability of matching IATA SHC = **`DGR`** (Hazardous Materials/Lithium Batteries). This combination forces `storage_type` = **`"Dangerous Goods"`** and `temp_range` = **`"Ambient"`**.
* **All Unlisted General Routes (e.g., `JED-LHR`):** Default to IATA SHC = **`GEN`** (General Cargo), forcing `storage_type` = **`"General"`** and `temp_range` = **`"Ambient"`**.

---

## 4. Complete Feature-by-Feature Specification & Value Scopes

| No. | Feature Name | Data Type | Value Scope (Allowed Range) | Functional Engineering Definition & Deterministic Generation Formula |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `timestamp` | Datetime | `2025-03-01 00:00:00` to `2026-02-28 23:00:00` | The primary temporal index. Follows strict chronological hourly increments (`YYYY-MM-DD HH:00:00`). |
| 2 | `flow_direction` | Categorical | `Import`, `Export` | Indicates the operational direction of freight travel. Controls whether the row follows arrival or departure scheduling logic. |
| 3 | `storage_type` | Categorical | `General`, `Cold Chain`, `Dangerous Goods`, `VIP` | Indicates the specific climate and physical security allocation inside the terminal. Sets the capacity boundary. |
| 4 | `route` | Categorical | Airport Pairs (e.g., `AMS-JED`, `JED-FRA`) | Tracks trade lane origins and destinations. Always ends in `-JED` for imports and begins with `JED-` for exports. |
| 5 | `iata_shc` | Categorical | `PER`, `PIL`, `DGR`, `VAL`, `GEN` | Standard IATA Special Handling Codes. Enforces strict physical destination mapping limits as outlined in Section 3. |
| 6 | `temp_range` | Categorical | `Ambient`, `2-8°C`, `15-25°C` | Tracks the specific micro-climate requirements of the storage space. Bounded strictly by the `iata_shc` type. |
| 7 | `aircraft_type` | Categorical | `Belly`, `Freighter` | Indicates the transport vehicle category. Dictates the scale and intensity of incoming or outgoing volume payloads. |
| 8 | `build_up_status` | Categorical | `Not Started`, `In Progress`, `Not Applicable` | Progress state for flight assembly. Forced to `"Not Applicable"` for all `Import` rows. For `Export` rows, it switches to `"In Progress"` when $\text{hours\_until\_departure} \le 6$. |
| 9 | `zone_capacity_m3` | Integer | `600`, `1000`, `1500`, `4500`, `5500` | A static physical volume capacity cap. Determined strictly by the combination of `storage_type` and `flow_direction`. |
| 10 | `uld_count` | Integer | $0$ to $250$ Units | The number of physical aviation containers occupying floor space. Derived directly via the structural footprint ratio: $\text{uld\_count} = \lfloor \frac{\text{occupied\_volume\_m3}}{32} \rfloor + \mathcal{N}(0, 1)$. |
| 11 | `expected_flight_volume_kg` | Integer | $1,000$ to $100,000 \text{ kg}$ | Inbound or outbound cargo weight derived from flight manifests. Bounded by `aircraft_type`: `Belly` $\in [1000, 5000]$, `Freighter` $\in [40000, 100000]$. |
| 12 | `hours_until_arrival` | Integer | $0$ to $24 \text{ Hours}$ | The countdown timer tracking incoming flights. Counts down sequentially to $0$, then resets to $24$ based on flight schedules. |
| 13 | `hours_until_departure` | Integer | $0$ to $24 \text{ Hours}$ | The countdown timer tracking outbound flights. Counts down sequentially to $0$, then resets to $24$ based on flight schedules. |
| 14 | `historical_dwell_lag_24h` | Float | $6.0$ to $55.0 \text{ Hours}$ | The baseline logistics clearing speed. Base constants: Cold Chain=$6\text{h}$, VIP=$12\text{h}$, General Export=$14\text{h}$, General Import=$26\text{h}$, DG=$38\text{h}$. Adds $+10\text{h}$ on weekends. |
| 15 | `congestion_index` | Float | $0.00$ to $1.00$ Ratio | Measures active terminal friction. Formulated as: $\text{congestion\_index} = 0.4(\frac{\text{uld\_count}}{\text{max\_zone\_uld}}) + 0.4(\text{occupancy\_rate\_lag1}) + 0.2(\text{seasonal\_modifier})$. |
| 16 | `forecasted_demand_next_24h` | Float | Continuous Volume Size | Calculated lookahead demand indicator. Sums up all manifest entries (`expected_flight_volume_kg`) scheduled across the next 24 hours $\times 0.0032$. |
| 17 | `forecasted_demand_next_48h` | Float | Continuous Volume Size | Calculated lookahead demand indicator. Sums up all manifest entries (`expected_flight_volume_kg`) scheduled across the next 48 hours $\times 0.0032$. |
| 18 | `forecasted_demand_next_72h` | Float | Continuous Volume Size | Calculated lookahead demand indicator. Sums up all manifest entries (`expected_flight_volume_kg`) scheduled across the next 72 hours $\times 0.0032$. |
| 19 | `occupied_volume_m3` | Float | $30.0$ to $5,450.0 \text{ m}^3$ | **The Core Metric Numerator.** Calculated deterministically from current inflows, outflows, and backlog parameters as detailed in Section 5. |
| 20 | `available_space_m3` | Float | Continuous Remainder | The remaining open volume capacity. Enforced by the exact balance identity: $\text{zone\_capacity\_m3} - \text{occupied\_volume\_m3}$. |
| 21 | **`occupancy_rate`** | Float | **$5.0\%$ to $100.0\%$** | **The Primary Modeling Target.** Continuous percentage representing physical storage saturation: $(\text{occupied\_volume\_m3} / \text{zone\_capacity\_m3}) \times 100$. |

---

## 5. Multi-Entity Feature Interaction & Advanced Combination Mechanics

The features in this dataset do not move independently. They operate within a closed network that simulates the actual movement of cargo through a terminal layout.

### A. The "Water Tank" Fluid Dynamics Framework
To model future capacity changes, the variables are grouped into four operational categories that simulate a fluid volume system:



1. **The Starting Point ($T = 0$):** `occupied_volume_m3` and `available_space_m3` establish the baseline storage state.
2. **The Inflow Faucets:** `expected_flight_volume_kg` combined with `hours_until_arrival` tells the engine how much cargo volume is flying toward the airport and exactly when it will land on the floor.
3. **The Outflow Drains:** `hours_until_departure` combined with `build_up_status` tells the engine when cargo must be pulled *out* of the warehouse storage racks and rolled onto the tarmac to be loaded onto airplanes, emptying out the indoor space.
4. **The Velocity Modifiers:** `congestion_index` and `historical_dwell_lag_24h` act as indicators of processing speed. If congestion is high ($>0.80$), the model simulates a clogged drain—meaning cargo moves slower and remains on the floor longer, keeping occupancy rates high.

### B. Inbound Wave Logic (`flow_direction` == "Import")
When an import freighter flight approaches the airport (`hours_until_arrival` $\le 4$), the system simulates preliminary staging activities. Forklifts begin clearing lanes and moving equipment. This triggers an automated, non-linear ramp-up in `uld_count` and `congestion_index` *before* the plane lands, followed by an immediate volume spike once the countdown timer hits $0$.

### C. Outbound Wave Logic (`flow_direction` == "Export")
For export operations, space usage is driven by flight departure windows. When `hours_until_departure` $\le 6$, the `build_up_status` switches from `"Not Started"` to `"In Progress"`. This causes cargo to move from static racks out into the open staging lanes, increasing floor space usage. Once the timer hits $0$ (flight departure), the cargo leaves the building, and the occupancy rate drops sharply.



### D. Macro Seasonality and Calendar Multipliers
To capture real-world operational cycles, the baseline volume equations are modified by two seasonal adjustments:
* **The Q4 Holiday Peak Rush (Nov 1 to Dec 31):** To simulate the intense global e-commerce winter peak, all inbound flight volumes and core base volumes are automatically scaled by a **$1.35\times$ multiplication penalty**.
* **The Weekend Customs Bottleneck (Friday 12:00 to Saturday 23:59):** Over the weekend, local customs clearances and cargo dispatch lines scale back operations by **$40\%$**. This causes cargo to sit idle longer, artificially increasing `historical_dwell_lag_24h` and driving up weekend occupancy rates, independent of actual flight frequencies.

### E. Gridlock Feedback Loop
When high volumes cause the `congestion_index` to breach **`0.80`**, a non-linear operational bottleneck loop activates:

$$\text{historical\_dwell\_lag\_24h}_{\text{new}} = \text{historical\_dwell\_lag\_24h}_{\text{base}} \times 1.4$$

This behavior simulates terminal gridlock, where high density slows down forklift movement and extends overall processing times, trapping the room in a high occupancy state until the backlog can be cleared.
