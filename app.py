import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import datetime
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Warehouse Space Utilization Forecasting",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING (THEME AGNOSTIC)
# =========================================================
st.markdown("""
<style>

.block-container {
    max-width: 95%;
    padding-top: 2rem;
}

div.stButton > button {
    width: 100% !important;
    font-weight: 600;
    height: 52px;
    font-size: 15px;
}

/* Uses theme's secondary background and dynamic borders/text */
.stMetric {
    border: 1px solid var(--border-color);
    padding: 15px;
    border-radius: 6px;
    background-color: var(--secondary-background-color);
}

/* Category header text dynamically adapts to light/dark modes */
.category-header {
    border-left: 3px solid #0288d1;
    padding-left: 10px;
    margin-top: 10px;
    margin-bottom: 15px;
    color: var(--text-color);
}

.status-box {
    padding: 15px;
    border-radius: 5px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL PACKAGE
# =========================================================
@st.cache_resource
def load_production_package():
    try:
        with open("jeddah_cargo_model_package.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Production artifact package not found.")
        st.stop()
    except Exception as e:
        st.error(f"Model package loading failed: {e}")
        st.stop()

pkg = load_production_package()

production_scaler = pkg["scaler"]
production_ohe = pkg["ohe"]
production_te = pkg["target_encoder"]
production_model = pkg["model"]

numerical_features = pkg["numerical_features"]
low_cardinality_categoricals = pkg["low_cardinality_categoricals"]
high_cardinality_categorical = pkg["high_cardinality_categorical"]

# =========================================================
# DOMAIN CONFIGURATION
# =========================================================
ROUTE_PROFILES = {
    "Import": ["AMS-JED", "FRA-JED", "CDG-JED", "NBO-JED", "PVG-JED", "HKG-JED", "DXB-JED", "DOH-JED", "LHR-JED"],
    "Export": ["JED-NBO", "JED-PVG", "JED-FRA", "JED-HKG", "JED-CDG", "JED-DXB", "JED-DOH", "JED-LHR", "JED-AMS"]
}

STORAGE_CAPACITIES = {
    ("General", "Import"): 5500,
    ("General", "Export"): 4500,
    ("Cold Chain", "Import"): 1500,
    ("Dangerous Goods", "Import"): 1000,
    ("VIP", "Export"): 600
}

VALID_COMBINATIONS = [
    {"flow_direction": "Import", "storage_type": "General", "shc": "GEN", "temp": "Ambient"},
    {"flow_direction": "Import", "storage_type": "Cold Chain", "shc": "PER", "temp": "2-8°C"},
    {"flow_direction": "Import", "storage_type": "Cold Chain", "shc": "PIL", "temp": "15-25°C"},
    {"flow_direction": "Import", "storage_type": "Dangerous Goods", "shc": "DGR", "temp": "Ambient"},
    {"flow_direction": "Export", "storage_type": "General", "shc": "GEN", "temp": "Ambient"},
    {"flow_direction": "Export", "storage_type": "VIP", "shc": "VAL", "temp": "Ambient"}
]

# =========================================================
# HEADER
# =========================================================
st.title("Warehouse Space Utilization Forecasting")
st.markdown("---")
st.subheader("Operational Parameter Input Console")

# =========================================================
# INPUT PANELS
# =========================================================
col_left, col_mid, col_right = st.columns(3)

# =========================================================
# COLUMN 1
# =========================================================
with col_left:
    st.markdown('<div class="category-header"><h4>Structural Footprint</h4></div>', unsafe_allow_html=True)
    
    flow_direction_input = st.selectbox("flow direction", ["Import", "Export"])

    if flow_direction_input == "Import":
        storage_type_input = st.selectbox("storage type", ["General", "Cold Chain", "Dangerous Goods"])
    else:
        storage_type_input = st.selectbox("storage type", ["General", "VIP"])

    matched_set = [c for c in VALID_COMBINATIONS if c["flow_direction"] == flow_direction_input and c["storage_type"] == storage_type_input]
    available_shcs = [m["shc"] for m in matched_set]
    iata_shc_input = st.selectbox("iata shc", available_shcs)

    final_match = [m for m in matched_set if m["shc"] == iata_shc_input][0]
    temp_range = final_match["temp"]

    st.text_input("temp range", value=temp_range, disabled=True)
    available_routes = ROUTE_PROFILES[flow_direction_input]
    route_input = st.selectbox("route", available_routes)
    aircraft_type = st.selectbox("aircraft type", ["Freighter", "Belly"])

# =========================================================
# COLUMN 2
# =========================================================
with col_mid:
    st.markdown('<div class="category-header"><h4>Floor Telemetry</h4></div>', unsafe_allow_html=True)

    if aircraft_type == "Freighter":
        expected_flight_volume_kg = st.slider("expected flight volume kg", 40000, 100000, 75000)
    else:
        expected_flight_volume_kg = st.slider("expected flight volume kg", 1000, 5000, 3000)

    if flow_direction_input == "Import":
        build_up_status_input = st.selectbox("build up status", ["Not Applicable"], disabled=True)
    else:
        build_up_status_input = st.selectbox("build up status", ["Not Started", "In Progress"])

    uld_count = st.slider("uld count", 0, 250, 155)
    hours_until_arrival = st.slider("hours until arrival", 0, 72, 12)
    hours_until_departure = st.slider("hours until departure", 0, 72, 18)
    historical_dwell_lag_24h = st.slider("historical dwell lag 24h", 6.0, 55.0, 36.0, step=0.1)

# =========================================================
# COLUMN 3
# =========================================================
with col_right:
    st.markdown('<div class="category-header"><h4>Chronological Context</h4></div>', unsafe_allow_html=True)

    forecast_horizon = st.slider("forecast horizon hours", 24, 72, 24, step=1)
    congestion_index = st.slider("congestion index", 0.0, 1.0, 0.8533, step=0.0001)

    forecasted_demand_next_24h = st.number_input("forecasted demand next 24h", value=269.95)
    forecasted_demand_next_48h = st.number_input("forecasted demand next 48h", value=539.90)
    forecasted_demand_next_72h = st.number_input("forecasted demand next 72h", value=809.86)

    selected_date = st.date_input("target date", datetime.date.today())
    selected_time = st.time_input("target time", datetime.datetime.now().time())

# =========================================================
# EXECUTE BUTTON
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
execute_analysis = st.button("RUN PREDICTIVE CAPACITY ANALYSIS", type="primary")
st.markdown("---")

# =========================================================
# PREDICTION EXECUTION
# =========================================================
if execute_analysis:
    try:
        base_datetime = datetime.datetime.combine(selected_date, selected_time)
        future_datetime = base_datetime + datetime.timedelta(hours=forecast_horizon)

        hour_of_day = future_datetime.hour
        day_of_week = future_datetime.weekday()
        month_of_year = future_datetime.month

        adjusted_congestion = min(1.0, congestion_index + (forecast_horizon * 0.0025))
        adjusted_arrival = max(0, hours_until_arrival - forecast_horizon)
        adjusted_departure = max(0, hours_until_departure - forecast_horizon)

        if forecast_horizon <= 24:
            adjusted_demand = forecasted_demand_next_24h
        elif forecast_horizon <= 48:
            adjusted_demand = forecasted_demand_next_48h
        else:
            adjusted_demand = forecasted_demand_next_72h

        input_data = {
            "uld_count": [float(uld_count)],
            "expected_flight_volume_kg": [float(expected_flight_volume_kg)],
            "hours_until_arrival": [float(adjusted_arrival)],
            "hours_until_departure": [float(adjusted_departure)],
            "historical_dwell_lag_24h": [float(historical_dwell_lag_24h)],
            "congestion_index": [float(adjusted_congestion)],
            "forecasted_demand_next_24h": [float(forecasted_demand_next_24h)],
            "forecasted_demand_next_48h": [float(forecasted_demand_next_48h)],
            "forecasted_demand_next_72h": [float(forecasted_demand_next_72h)],
            "hour_of_day": [int(hour_of_day)],
            "day_of_week": [int(day_of_week)],
            "month_of_year": [int(month_of_year)],
            "flow_direction": [flow_direction_input],
            "storage_type": [storage_type_input],
            "iata_shc": [iata_shc_input],
            "build_up_status": [build_up_status_input],
            "route": [route_input]
        }

        df_input = pd.DataFrame(input_data)

        low_cardinality_headers = production_ohe.get_feature_names_out(low_cardinality_categoricals)
        scaled_num = pd.DataFrame(production_scaler.transform(df_input[numerical_features]), columns=numerical_features)
        scaled_ohe = pd.DataFrame(production_ohe.transform(df_input[low_cardinality_categoricals]), columns=low_cardinality_headers)
        scaled_te = production_te.transform(df_input[high_cardinality_categorical])

        X_inference = pd.concat([scaled_num, scaled_ohe, scaled_te], axis=1)

        predicted_occupancy_rate = production_model.predict(X_inference)[0]
        predicted_occupancy_rate = np.clip(predicted_occupancy_rate, 5.0, 100.0)

        zone_capacity_m3 = STORAGE_CAPACITIES[(storage_type_input, flow_direction_input)]
        computed_occupied_m3 = (predicted_occupancy_rate / 100) * zone_capacity_m3
        computed_available_m3 = zone_capacity_m3 - computed_occupied_m3

        # =================================================
        # RESULT SECTION
        # =================================================
        st.header("Real-Time Operational Space Diagnostics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Total Structural Capacity", f"{zone_capacity_m3:,.0f} m3")
        with metric_col2:
            st.metric("Predicted Occupied Volume", f"{computed_occupied_m3:,.2f} m3")
        with metric_col3:
            st.metric("Available Net Open Space", f"{computed_available_m3:,.2f} m3")

        st.markdown("<br>", unsafe_allow_html=True)

        if predicted_occupancy_rate >= 90:
            gauge_color = "#dc3545"
            st.error(f"Critical saturation threshold detected ({predicted_occupancy_rate:.2f}%).")
        elif predicted_occupancy_rate >= 75:
            gauge_color = "#ffc107"
            st.warning(f"High congestion environment predicted ({predicted_occupancy_rate:.2f}%).")
        else:
            gauge_color = "#28a745"
            st.success(f"Operationally stable utilization forecast ({predicted_occupancy_rate:.2f}%).")

        # =================================================
        # GAUGE VISUALIZATION (THEME RESPONSIVE)
        # =================================================
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_occupancy_rate,
                number={
                    "valueformat": ".2f",
                    "suffix": "%",
                    "font": {"size": 36, "color": gauge_color}
                },
                title={"text": "Forecasted Space Occupancy Rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    # "bgcolor" removed to seamlessly pick up light/dark chart backgrounds
                    "steps": [
                        {"range": [0, 75], "color": "rgba(40,167,69,0.15)"},
                        {"range": [75, 90], "color": "rgba(255,193,7,0.15)"},
                        {"range": [90, 100], "color": "rgba(220,53,69,0.15)"}
                    ]
                }
            )
        )

        fig_gauge.update_layout(
            margin=dict(l=30, r=30, t=60, b=20),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # =================================================
        # FORECAST SUMMARY
        # =================================================
        st.subheader("Forecast Summary")

        summary_df = pd.DataFrame({
            "Parameter": [
                "Forecast Horizon", "Target Forecast Timestamp", "Flow Direction", 
                "Storage Type", "Cargo Category", "Selected Route", "Aircraft Type", "Adjusted Congestion Index"
            ],
            "Value": [
                f"{forecast_horizon} Hours", future_datetime.strftime("%Y-%m-%d %H:%M"), flow_direction_input,
                storage_type_input, iata_shc_input, route_input, aircraft_type, str(round(adjusted_congestion, 4))
            ]
        })

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Inference execution failed: {e}")
