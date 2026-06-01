import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
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
# CUSTOM STYLING
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

.stMetric {
    border: 1px solid var(--border-color);
    padding: 15px;
    border-radius: 6px;
    background-color: var(--secondary-background-color);
}

.category-header {
    border-left: 3px solid #0288d1;
    padding-left: 10px;
    margin-top: 10px;
    margin-bottom: 15px;
    color: var(--text-color);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL PACKAGE
# =========================================================
@st.cache_resource
def load_production_package():

    try:

        with open(
            "jeddah_cargo_model_package.pkl",
            "rb"
        ) as f:

            return pickle.load(f)

    except FileNotFoundError:

        st.error(
            "Production artifact package not found."
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Model package loading failed: {e}"
        )

        st.stop()

pkg = load_production_package()

production_scaler = pkg["scaler"]
production_ohe = pkg["ohe"]
production_te = pkg["target_encoder"]
production_model = pkg["model"]

numerical_features = pkg["numerical_features"]

low_cardinality_categoricals = pkg[
    "low_cardinality_categoricals"
]

high_cardinality_categorical = pkg[
    "high_cardinality_categorical"
]

# =========================================================
# LOAD ANALYTICS DATASET
# =========================================================
@st.cache_data
def load_analytics_data():

    try:

        df = pd.read_csv(
            "jeddah_air_cargo_occupancy_master.csv"
        )

        # auto detect timestamp column
        possible_timestamp_cols = [
            "timestamp",
            "datetime",
            "date_time",
            "event_timestamp"
        ]

        for col in possible_timestamp_cols:

            if col in df.columns:

                df[col] = pd.to_datetime(df[col])

                if col != "timestamp":
                    df.rename(
                        columns={col: "timestamp"},
                        inplace=True
                    )

                break

        return df

    except Exception as e:

        st.error(
            f"Analytics dataset loading failed: {e}"
        )

        st.stop()

analytics_df = load_analytics_data()

# =========================================================
# AUTO DETECT OCCUPANCY COLUMN
# =========================================================
occupancy_column = None

possible_occupancy_columns = [

    "predicted_occupancy_rate_24h",
    "warehouse_occupancy_percent",
    "occupancy_rate",
    "occupancy_percent"
]

for col in possible_occupancy_columns:

    if col in analytics_df.columns:

        occupancy_column = col
        break

# =========================================================
# DOMAIN CONFIGURATION
# =========================================================
ROUTE_PROFILES = {

    "Import": [
        "AMS-JED",
        "FRA-JED",
        "CDG-JED",
        "NBO-JED",
        "PVG-JED",
        "HKG-JED",
        "DXB-JED",
        "DOH-JED",
        "LHR-JED"
    ],

    "Export": [
        "JED-NBO",
        "JED-PVG",
        "JED-FRA",
        "JED-HKG",
        "JED-CDG",
        "JED-DXB",
        "JED-DOH",
        "JED-LHR",
        "JED-AMS"
    ]
}

STORAGE_CAPACITIES = {

    ("General", "Import"): 5500,
    ("General", "Export"): 4500,
    ("Cold Chain", "Import"): 1500,
    ("Dangerous Goods", "Import"): 1000,
    ("VIP", "Export"): 600
}

VALID_COMBINATIONS = [

    {
        "flow_direction": "Import",
        "storage_type": "General",
        "shc": "GEN",
        "temp": "Ambient"
    },

    {
        "flow_direction": "Import",
        "storage_type": "Cold Chain",
        "shc": "PER",
        "temp": "2-8°C"
    },

    {
        "flow_direction": "Import",
        "storage_type": "Cold Chain",
        "shc": "PIL",
        "temp": "15-25°C"
    },

    {
        "flow_direction": "Import",
        "storage_type": "Dangerous Goods",
        "shc": "DGR",
        "temp": "Ambient"
    },

    {
        "flow_direction": "Export",
        "storage_type": "General",
        "shc": "GEN",
        "temp": "Ambient"
    },

    {
        "flow_direction": "Export",
        "storage_type": "VIP",
        "shc": "VAL",
        "temp": "Ambient"
    }
]

# =========================================================
# HEADER
# =========================================================
st.title(
    "Warehouse Space Utilization Forecasting"
)

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs([
    "Forecast Prediction",
    "Analytics Dashboard"
])

# =========================================================
# TAB 1 : PREDICTION
# =========================================================
with tab1:

    st.subheader(
        "Operational Parameter Input Console"
    )

    col_left, col_mid, col_right = st.columns(3)

    # =====================================================
    # COLUMN 1
    # =====================================================
    with col_left:

        st.markdown(
            '<div class="category-header"><h4>Structural Footprint</h4></div>',
            unsafe_allow_html=True
        )

        flow_direction_input = st.selectbox(
            "flow direction",
            ["Import", "Export"]
        )

        if flow_direction_input == "Import":

            storage_type_input = st.selectbox(
                "storage type",
                [
                    "General",
                    "Cold Chain",
                    "Dangerous Goods"
                ]
            )

        else:

            storage_type_input = st.selectbox(
                "storage type",
                [
                    "General",
                    "VIP"
                ]
            )

        matched_set = [

            c for c in VALID_COMBINATIONS

            if c["flow_direction"]
            == flow_direction_input

            and

            c["storage_type"]
            == storage_type_input
        ]

        available_shcs = [
            m["shc"] for m in matched_set
        ]

        iata_shc_input = st.selectbox(
            "iata shc",
            available_shcs
        )

        final_match = [

            m for m in matched_set

            if m["shc"]
            == iata_shc_input
        ][0]

        temp_range = final_match["temp"]

        st.text_input(
            "temp range",
            value=temp_range,
            disabled=True
        )

        available_routes = ROUTE_PROFILES[
            flow_direction_input
        ]

        route_input = st.selectbox(
            "route",
            available_routes
        )

        aircraft_type = st.selectbox(
            "aircraft type",
            ["Freighter", "Belly"]
        )

    # =====================================================
    # COLUMN 2
    # =====================================================
    with col_mid:

        st.markdown(
            '<div class="category-header"><h4>Floor Telemetry</h4></div>',
            unsafe_allow_html=True
        )

        if aircraft_type == "Freighter":

            expected_flight_volume_kg = st.slider(
                "expected flight volume kg",
                40000,
                100000,
                75000
            )

        else:

            expected_flight_volume_kg = st.slider(
                "expected flight volume kg",
                1000,
                5000,
                3000
            )

        if flow_direction_input == "Import":

            build_up_status_input = st.selectbox(
                "build up status",
                ["Not Applicable"],
                disabled=True
            )

        else:

            build_up_status_input = st.selectbox(
                "build up status",
                [
                    "Not Started",
                    "In Progress"
                ]
            )

        uld_count = st.slider(
            "uld count",
            0,
            250,
            155
        )

        hours_until_arrival = st.slider(
            "hours until arrival",
            0,
            72,
            12
        )

        hours_until_departure = st.slider(
            "hours until departure",
            0,
            72,
            18
        )

        historical_dwell_lag_24h = st.slider(
            "historical dwell lag 24h",
            6.0,
            55.0,
            36.0,
            step=0.1
        )

    # =====================================================
    # COLUMN 3
    # =====================================================
    with col_right:

        st.markdown(
            '<div class="category-header"><h4>Chronological Context</h4></div>',
            unsafe_allow_html=True
        )

        forecast_horizon = st.slider(
            "forecast horizon hours",
            24,
            72,
            24,
            step=1
        )

        congestion_index = st.slider(
            "congestion index",
            0.0,
            1.0,
            0.8533,
            step=0.0001
        )

        forecasted_demand_next_24h = st.number_input(
            "forecasted demand next 24h",
            value=269.95
        )

        forecasted_demand_next_48h = st.number_input(
            "forecasted demand next 48h",
            value=539.90
        )

        forecasted_demand_next_72h = st.number_input(
            "forecasted demand next 72h",
            value=809.86
        )

        selected_date = st.date_input(
            "target date",
            datetime.date.today()
        )

        selected_time = st.time_input(
            "target time",
            datetime.datetime.now().time()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    execute_analysis = st.button(
        "RUN PREDICTIVE CAPACITY ANALYSIS",
        type="primary"
    )

    st.markdown("---")

    # =====================================================
    # PREDICTION EXECUTION
    # =====================================================
    if execute_analysis:

        try:

            base_datetime = datetime.datetime.combine(
                selected_date,
                selected_time
            )

            future_datetime = (
                base_datetime +
                datetime.timedelta(
                    hours=forecast_horizon
                )
            )

            hour_of_day = future_datetime.hour
            day_of_week = future_datetime.weekday()
            month_of_year = future_datetime.month

            adjusted_congestion = min(
                1.0,
                congestion_index +
                (
                    forecast_horizon * 0.0025
                )
            )

            adjusted_arrival = max(
                0,
                hours_until_arrival -
                forecast_horizon
            )

            adjusted_departure = max(
                0,
                hours_until_departure -
                forecast_horizon
            )

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

            low_cardinality_headers = (
                production_ohe.get_feature_names_out(
                    low_cardinality_categoricals
                )
            )

            scaled_num = pd.DataFrame(
                production_scaler.transform(
                    df_input[
                        numerical_features
                    ]
                ),
                columns=numerical_features
            )

            scaled_ohe = pd.DataFrame(
                production_ohe.transform(
                    df_input[
                        low_cardinality_categoricals
                    ]
                ),
                columns=low_cardinality_headers
            )

            scaled_te = production_te.transform(
                df_input[
                    high_cardinality_categorical
                ]
            )

            X_inference = pd.concat([
                scaled_num,
                scaled_ohe,
                scaled_te
            ], axis=1)

            predicted_occupancy_rate = (
                production_model.predict(
                    X_inference
                )[0]
            )

            predicted_occupancy_rate = np.clip(
                predicted_occupancy_rate,
                5.0,
                100.0
            )

            zone_capacity_m3 = STORAGE_CAPACITIES[
                (
                    storage_type_input,
                    flow_direction_input
                )
            ]

            computed_occupied_m3 = (
                predicted_occupancy_rate / 100
            ) * zone_capacity_m3

            computed_available_m3 = (
                zone_capacity_m3 -
                computed_occupied_m3
            )

            st.header(
                "Real-Time Operational Space Diagnostics"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:

                st.metric(
                    "Total Structural Capacity",
                    f"{zone_capacity_m3:,.0f} m3"
                )

            with metric_col2:

                st.metric(
                    "Predicted Occupied Volume",
                    f"{computed_occupied_m3:,.2f} m3"
                )

            with metric_col3:

                st.metric(
                    "Available Net Open Space",
                    f"{computed_available_m3:,.2f} m3"
                )

            fig_gauge = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=predicted_occupancy_rate,

                    number={
                        "valueformat": ".2f",
                        "suffix": "%"
                    },

                    title={
                        "text":
                        "Forecasted Space Occupancy Rate"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "color": "#0288d1"
                        }
                    }
                )
            )

            fig_gauge.update_layout(
                height=350
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Inference execution failed: {e}"
            )

# =========================================================
# TAB 2 : ANALYTICS DASHBOARD
# =========================================================
with tab2:

    st.subheader(
        "Warehouse Operations Analytics Dashboard"
    )

    if occupancy_column is None:

        st.error(
            "No occupancy column found in dataset."
        )

        st.write(
            analytics_df.columns.tolist()
        )

    else:

        # =================================================
        # FILTERS
        # =================================================
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            if 'storage_type' in analytics_df.columns:

                storage_filter = st.multiselect(
                    "Filter Storage Type",
                    options=sorted(
                        analytics_df[
                            'storage_type'
                        ].dropna().unique().tolist()
                    ),
                    default=sorted(
                        analytics_df[
                            'storage_type'
                        ].dropna().unique().tolist()
                    )
                )

            else:

                storage_filter = []

        with filter_col2:

            if 'iata_shc' in analytics_df.columns:

                cargo_filter = st.multiselect(
                    "Filter Cargo Type",
                    options=sorted(
                        analytics_df[
                            'iata_shc'
                        ].dropna().unique().tolist()
                    ),
                    default=sorted(
                        analytics_df[
                            'iata_shc'
                        ].dropna().unique().tolist()
                    )
                )

            else:

                cargo_filter = []

        filtered_df = analytics_df.copy()

        if (
            'storage_type' in filtered_df.columns
            and
            len(storage_filter) > 0
        ):

            filtered_df = filtered_df[
                filtered_df['storage_type']
                .isin(storage_filter)
            ]

        if (
            'iata_shc' in filtered_df.columns
            and
            len(cargo_filter) > 0
        ):

            filtered_df = filtered_df[
                filtered_df['iata_shc']
                .isin(cargo_filter)
            ]

        st.markdown("---")

        # =================================================
        # KPI CARDS
        # =================================================
        k1, k2, k3, k4 = st.columns(4)

        with k1:

            st.metric(
                "Average Occupancy",
                f"{filtered_df[occupancy_column].mean():.2f}%"
            )

        with k2:

            if 'historical_dwell_lag_24h' in filtered_df.columns:

                st.metric(
                    "Average Dwell Lag",
                    f"{filtered_df['historical_dwell_lag_24h'].mean():.1f} hrs"
                )

            else:

                st.metric(
                    "Average Dwell Lag",
                    "N/A"
                )

        with k3:

            if 'congestion_index' in filtered_df.columns:

                st.metric(
                    "Average Congestion",
                    f"{filtered_df['congestion_index'].mean():.2f}"
                )

            else:

                st.metric(
                    "Average Congestion",
                    "N/A"
                )

        with k4:

            if 'expected_flight_volume_kg' in filtered_df.columns:

                st.metric(
                    "Average Flight Volume",
                    f"{filtered_df['expected_flight_volume_kg'].mean():,.0f} kg"
                )

            else:

                st.metric(
                    "Average Flight Volume",
                    "N/A"
                )

        st.markdown("---")

        # =================================================
        # CHART ROW 1
        # =================================================
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:

            st.markdown(
                "### Average Occupancy by Storage Type"
            )

            if 'storage_type' in filtered_df.columns:

                storage_summary = (
                    filtered_df.groupby(
                        'storage_type'
                    )[
                        occupancy_column
                    ]
                    .mean()
                    .reset_index()
                )

                fig_storage = px.bar(
                    storage_summary,
                    x='storage_type',
                    y=occupancy_column,
                    text_auto=True
                )

                fig_storage.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_storage,
                    use_container_width=True
                )

        with row1_col2:

            st.markdown(
                "### Cargo Category Distribution"
            )

            if 'iata_shc' in filtered_df.columns:

                cargo_summary = (
                    filtered_df.groupby(
                        'iata_shc'
                    )
                    .size()
                    .reset_index(name='Count')
                )

                fig_cargo = px.bar(
                    cargo_summary,
                    x='iata_shc',
                    y='Count',
                    text_auto=True
                )

                fig_cargo.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_cargo,
                    use_container_width=True
                )

        st.markdown("---")

        # =================================================
        # CHART ROW 2
        # =================================================
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:

            st.markdown(
                "### Congestion by Storage Type"
            )

            if (
                'storage_type' in filtered_df.columns
                and
                'congestion_index' in filtered_df.columns
            ):

                congestion_summary = (
                    filtered_df.groupby(
                        'storage_type'
                    )[
                        'congestion_index'
                    ]
                    .mean()
                    .reset_index()
                )

                fig_congestion = px.bar(
                    congestion_summary,
                    x='storage_type',
                    y='congestion_index',
                    text_auto=True
                )

                fig_congestion.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_congestion,
                    use_container_width=True
                )

        with row2_col2:

            st.markdown(
                "### Flight Volume by Cargo Type"
            )

            if (
                'iata_shc' in filtered_df.columns
                and
                'expected_flight_volume_kg' in filtered_df.columns
            ):

                volume_summary = (
                    filtered_df.groupby(
                        'iata_shc'
                    )[
                        'expected_flight_volume_kg'
                    ]
                    .mean()
                    .reset_index()
                )

                fig_volume = px.bar(
                    volume_summary,
                    x='iata_shc',
                    y='expected_flight_volume_kg',
                    text_auto=True
                )

                fig_volume.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig_volume,
                    use_container_width=True
                )

        st.markdown("---")

        # =================================================
        # TREND GRAPH
        # =================================================
        st.markdown(
            "### Daily Occupancy Trend"
        )

        if 'timestamp' in filtered_df.columns:

            trend_df = (
                filtered_df.groupby(
                    filtered_df[
                        'timestamp'
                    ].dt.date
                )[
                    occupancy_column
                ]
                .mean()
                .reset_index()
            )

            fig_trend = px.line(
                trend_df,
                x='timestamp',
                y=occupancy_column,
                markers=True
            )

            fig_trend.update_layout(
                height=400
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True
            )
