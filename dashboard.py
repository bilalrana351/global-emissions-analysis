import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global CO2 Analysis 2023",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stat-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #2C3E50;
            margin: 5px 0;
        }
        .stat-label {
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .block-container {
            padding-top: 1rem;
        }
        div.stButton > button:first-child {
            background-color: #f0f2f6;
            color: #31333F;
            border: 1px solid #d6d6d6;
        }
        div.stButton > button:active {
            background-color: #e0e2e6;
        }
        h3 {
            font-size: 22px !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
# We use a sidebar radio to toggle pages. 
# We maintain the "selected_country" in session state so it persists across pages.
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = None

page = st.sidebar.radio("Navigate Dashboard", ["Current Status (Page 1)", "Future Outlook (Page 2)"])
st.sidebar.markdown("---")

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    # Ensure this CSV file exists in your directory
    df = pd.read_csv('cleaned_co2_data_enriched.csv')
    
    # Filter for 2023
    df_2023 = df[df['Year'] == 2023].copy()
    
    # Remove Aggregates
    aggregates = ['World', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania',
                  'High-income countries', 'Upper-middle-income countries', 
                  'Lower-middle-income countries', 'Low-income countries', 'European Union (27)']
    df_2023 = df_2023[~df_2023['Country'].isin(aggregates)]
    
    # Clean Data
    df_2023 = df_2023.dropna(subset=['Income Group', 'Population', 'CO2 Emissions'])
    
    # Calculate Total Warming Column (Sum of CO2, Methane, N2O)
    df_2023['Calculated_Total_Warming'] = (
        df_2023['Temperature Change From CO2'].fillna(0) + 
        df_2023['Temperature Change From Methane'].fillna(0) + 
        df_2023['Temperature Change From Nitrous Oxide'].fillna(0)
    )
    
    return df_2023

@st.cache_data
def load_historical_data():
    df = pd.read_csv('cleaned_co2_data_enriched.csv')
    aggregates = ['World', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania',
                  'High-income countries', 'Upper-middle-income countries', 
                  'Lower-middle-income countries', 'Low-income countries', 'European Union (27)']
    df = df[~df['Country'].isin(aggregates)]
    return df

# =========================================================
# PAGE 1: CURRENT STATUS (Original Code)
# =========================================================
if page == "Current Status (Page 1)":
    df = load_data()

    # --- GLOBAL CALCULATIONS ---
    total_global_co2 = df['CO2 Emissions'].sum()
    total_global_pop = df['Population'].sum()
    total_global_warming = df['Calculated_Total_Warming'].sum()

    # Pre-calculate Global Income Group Data
    income_order = ['High Income', 'Upper-Middle Income', 'Lower-Middle Income', 'Low Income']
    income_df = df.groupby('Income Group')[['CO2 Emissions', 'Population']].sum().reset_index()
    income_df['CO2_Share'] = (income_df['CO2 Emissions'] / total_global_co2) * 100
    income_df['Pop_Share'] = (income_df['Population'] / total_global_pop) * 100
    income_df['Income Group'] = pd.Categorical(income_df['Income Group'], categories=income_order, ordered=True)
    income_df = income_df.sort_values('Income Group')

    # --- SIDEBAR INFO ---
    with st.sidebar:
        st.header("About Dashboard")
        st.markdown("""
        **Goal:** Analyze a country's climate impact through four key lenses:
        
        1.  **Inequality:** Actual CO2 Emissions vs. Fair Share of CO2 Emissions by Population.
        2.  **Scale:** Contribution to the annual global CO2 total.
        3.  **Sources:** The fuel mix (What are countries burning?).
        4.  **Legacy:** **Contribution of Countries to Global Warming** (Historical Debt).
        
        **Instructions:**
        1. Click a country on the map.
        2. Charts will update automatically.
        3. Use the Reset button to go back.
        """)
        st.markdown("---")
        st.markdown("Designed for 2023 Data Analysis.")

    # --- MAIN TITLE ---
    st.title("Global CO2 Inequality Dashboard (2023)")
    st.markdown("Select a country on the map to visualize its full climate story.")
    st.divider()

    # --- LAYOUT GRID ---
    col_map, col_viz = st.columns([1.4, 1.1], gap="large")

    # =========================================================
    # COLUMN 1: INTERACTIVE MAP & RESET BUTTON
    # =========================================================
    with col_map:
        # HEADER WITH RESET BUTTON
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("1. Global CO2 Emissions Map")
        with c2:
            if st.button("Reset Dashboard", use_container_width=True):
                st.session_state.selected_country = None
                st.rerun()

        # MAP
        fig_map = px.choropleth(
            df,
            locations="ISO Country Code",
            color="CO2 Emissions",
            hover_name="Country",
            hover_data={"CO2 Emissions": ":.1f", "Income Group": True, "ISO Country Code": False},
            color_continuous_scale="Reds",
            range_color=(0, 1000), 
            projection="equirectangular", 
        )
        
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Mt CO2"),
            height=550,
            clickmode='event+select',
            dragmode='pan',
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                showframe=False,
                showcoastlines=True,
                projection_scale=1.1
            )
        )
        
        selection = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points")
        
        if selection and len(selection['selection']['points']) > 0:
            idx = selection['selection']['points'][0]['point_index']
            try:
                st.session_state.selected_country = df.iloc[idx]['Country']
            except:
                pass

    # =========================================================
    # COLUMN 2: ANALYSIS DASHBOARD
    # =========================================================
    with col_viz:
        target = st.session_state.selected_country
        
        if target:
            # --- DRILL DOWN VIEW ---
            st.subheader(f"2. Analysis: {target}")
            
            # 1. Fetch Data
            c_row = df[df['Country'] == target].iloc[0]
            c_income = c_row['Income Group']
            c_co2 = c_row['CO2 Emissions']
            c_pop = c_row['Population']
            c_warming = c_row['Calculated_Total_Warming']
            
            # 2. Context Calculation
            grp_df = df[df['Income Group'] == c_income]
            grp_co2 = grp_df['CO2 Emissions'].sum()
            grp_pop = grp_df['Population'].sum()
            
            # 3. Share Calculation
            share_co2_group = (c_co2 / grp_co2) * 100
            share_pop_group = (c_pop / grp_pop) * 100
            share_co2_global = (c_co2 / total_global_co2) * 100
            share_warming_global = (c_warming / total_global_warming) * 100
            
            # 4. Custom Stat Cards
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""<div class="stat-card"><div class="stat-label">Income Group</div><div class="stat-value" style="font-size:18px;">{c_income}</div></div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class="stat-card"><div class="stat-label">CO2 Emissions</div><div class="stat-value">{c_co2:,.1f}<br><small style='font-size:12px; color:#888'>Mt CO2</small></div></div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""<div class="stat-card"><div class="stat-label">Global CO2 Share</div><div class="stat-value">{share_co2_global:.2f}%<br><small style='font-size:12px; color:#888'>of World Total</small></div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # --- ROW 1 CHARTS: Inequality & Scale ---
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                # CHART A: FAIRNESS CHECK
                rest_co2 = 100 - share_co2_group
                rest_pop = 100 - share_pop_group
                
                # --- FIX FOR X-AXIS LABELS ---
                def wrap_label(label, width=12):
                    return "<br>".join(textwrap.wrap(label, width=width))

                label_target = wrap_label(target, 15)
                label_rest = wrap_label(f"Rest of {c_income} countries", 15)
                
                plot_data = pd.DataFrame({
                    'Entity': [label_target, label_rest],
                    'Actual CO2 Share': [share_co2_group, rest_co2],
                    'Fair CO2 Share by Population': [share_pop_group, rest_pop] # UPDATED LABEL
                })
                plot_melt = plot_data.melt(id_vars='Entity', var_name='Metric', value_name='Percentage')
                
                fig_bar = px.bar(
                    plot_melt, x='Entity', y='Percentage', color='Metric',
                    barmode='group', text_auto='.1f',
                    color_discrete_map={'Actual CO2 Share': '#E74C3C', 'Fair CO2 Share by Population': '#2ECC71'} # UPDATED LABEL
                )
                
                # UPDATED LAYOUT FOR SPACING
                fig_bar.update_layout(
                    title=dict(text=f"<b>CO2 Inequality Check</b><br><span style='font-size:12px'>vs. {c_income} Peers</span>", x=0.5, xanchor='center'),
                    yaxis_title="% of Group CO2 Total", xaxis_title="",
                    legend=dict(orientation="h", y=-0.7, x=0.5, xanchor='center', title=""), # Moved Legend Lower
                    height=350, # Increased Height
                    template="simple_white", 
                    margin=dict(t=50, b=130, l=10, r=10), # Increased Bottom Margin
                    xaxis=dict(tickangle=0)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with row1_col2:
                # CHART B: GLOBAL FOOTPRINT
                donut_data = pd.DataFrame({
                    'Region': [target, 'Rest of World'],
                    'Emissions': [c_co2, total_global_co2 - c_co2]
                })
                fig_donut = px.pie(
                    donut_data, values='Emissions', names='Region',
                    hole=0.6, color='Region',
                    color_discrete_map={target: '#E74C3C', 'Rest of World': '#ecf0f1'}
                )
                fig_donut.update_traces(textinfo='none', hoverinfo='label+percent+value')
                fig_donut.add_annotation(
                    text=f"<b>{share_co2_global:.1f}%</b>", 
                    x=0.5, y=0.5, showarrow=False, font_size=20, font_color="#2C3E50"
                )
                fig_donut.update_layout(
                    title=dict(text=f"<b>Global CO2 Footprint</b><br><span style='font-size:12px'>Share of Global CO2 Emissions</span>", x=0.5, xanchor='center'),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor='center'),
                    height=350, template="simple_white", margin=dict(t=50, b=50, l=10, r=10)
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # --- ROW 2 CHARTS: Sources & Warming ---
            st.markdown("")
            row2_col1, row2_col2 = st.columns(2)

            with row2_col1:
                # CHART C: SOURCES (Donut)
                source_cols = ['Coal CO2 Emissions', 'Oil CO2 Emissions', 'Gas CO2 Emissions', 
                               'Cement CO2 Emissions', 'Flaring CO2 Emissions', 'Land Use Change CO2']
                source_labels = {'Coal CO2 Emissions': 'Coal', 'Oil CO2 Emissions': 'Oil', 
                                 'Gas CO2 Emissions': 'Gas', 'Cement CO2 Emissions': 'Cement', 
                                 'Flaring CO2 Emissions': 'Flaring', 'Land Use Change CO2': 'Land Use Change'}
                
                country_sources = c_row[source_cols].fillna(0)
                source_df = pd.DataFrame({
                    'Source': [source_labels[c] for c in source_cols],
                    'Value': country_sources.values
                })
                source_df = source_df[source_df['Value'] > 0]
                
                if not source_df.empty:
                    fig_sources = px.pie(
                        source_df, values='Value', names='Source', 
                        hole=0.6, 
                        color='Source',
                        color_discrete_map={
                            'Coal': '#2c3e50', 'Oil': '#e67e22', 'Gas': '#f1c40f',
                            'Cement': '#95a5a6', 'Flaring': '#8e44ad', 'Land Use Change': '#27ae60'
                        }
                    )
                    fig_sources.update_traces(textposition='inside', textinfo='percent+label')
                    
                    fig_sources.update_layout(
                        title=dict(text=f"<b>CO2 Emissions by Source</b><br><span style='font-size:12px'>What is {target} burning?</span>", x=0.5, xanchor='center'),
                        showlegend=True,
                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor='center'),
                        height=450, 
                        template="simple_white", 
                        margin=dict(t=50, b=50, l=10, r=10) 
                    )
                    st.plotly_chart(fig_sources, use_container_width=True)
                else:
                    st.info("No detailed source data available.")

            with row2_col2:
                # CHART D: TEMPERATURE CONTRIBUTION
                temp_cols = {'Temperature Change From CO2': 'CO2', 
                             'Temperature Change From Methane': 'Methane', 
                             'Temperature Change From Nitrous Oxide': 'N2O'}
                
                temp_values = []
                for col, name in temp_cols.items():
                    val = c_row.get(col, 0)
                    temp_values.append({'Gas': name, 'Value': val})
                
                temp_df = pd.DataFrame(temp_values)
                
                if temp_df['Value'].sum() > 0:
                    fig_temp = px.bar(
                        temp_df, x='Value', y='Gas', orientation='h', 
                        color='Gas', text_auto='.3f',
                        color_discrete_map={
                            'CO2': '#2c3e50',    
                            'Methane': '#e67e22', 
                            'N2O': '#27ae60'      
                        }
                    )
                    
                    fig_temp.update_layout(
                        title=dict(text=f"<b>Global Warming Impact</b><br><span style='font-size:12px'>Responsible for <b style='color:#e74c3c'>{share_warming_global:.2f}%</b> of Global Warming</span>", x=0.5, xanchor='center'),
                        xaxis_title="Degrees Celsius (°C) Contribution", yaxis_title="",
                        showlegend=False,
                        height=450, # Matched Height with Chart C
                        template="simple_white", 
                        margin=dict(t=50, b=20, l=10, r=10)
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                else:
                    st.info("No temperature data available.")

            # Insight Box
            gap = share_co2_group - share_pop_group
            if gap > 0:
                msg = f"⚠️ <b>Overshoot: {target} emits more CO2 than its fair share compared to other {c_income} countries.</b>"
                style = "background-color: #fceceb; border-left: 5px solid #e74c3c; padding: 10px;"
            else:
                msg = f"✅ <b>Sustainable: {target} emits less CO2 than its population share compared to other {c_income} countries.</b>"
                style = "background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 10px;"
                
            st.markdown(f"<div style='{style}'>{msg}</div>", unsafe_allow_html=True)

        else:
            # --- DEFAULT GLOBAL VIEW ---
            st.subheader("2. Global CO2 Overview")
            st.info("Click on a country in the map to see its detailed stats.")
            
            # Fix labels for global view
            income_df_plot = income_df.copy()
            income_df_plot['Income Group'] = income_df_plot['Income Group'].astype(str) + " Countries"

            fig_global = go.Figure()
            fig_global.add_trace(go.Bar(
                x=income_df_plot['Income Group'], y=income_df_plot['CO2_Share'],
                name='Actual CO2 Emissions', marker_color='#E74C3C',
                text=income_df_plot['CO2_Share'].apply(lambda x: f"{x:.1f}%"), textposition='auto'
            ))
            fig_global.add_trace(go.Bar(
                x=income_df_plot['Income Group'], y=income_df_plot['Pop_Share'],
                name='Fair CO2 Share by Population',
                marker_color='#2ECC71',
                text=income_df_plot['Pop_Share'].apply(lambda x: f"{x:.1f}%"), textposition='auto'
            ))
            fig_global.update_layout(
                title="Countries Income Groups: Actual vs. Fair CO2 Emissions Based on Population",
                barmode='group', yaxis_title="% of Global CO2 Total", xaxis_title="",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
                height=450, template="simple_white"
            )
            st.plotly_chart(fig_global, use_container_width=True)

# =========================================================
# PAGE 2: FUTURE OUTLOOK (Actionable Insights)
# =========================================================
elif page == "Future Outlook (Page 2)":
    
    # 1. Title & Header
    st.title("Strategic Climate Action Planner")
    st.markdown("Forecasting emissions scenarios to 2050 to identify concrete decarbonization pathways.")
    st.divider()

    # 2. Country Selector for Page 2
    # Syncs with Page 1 selection if available
    df_hist = load_historical_data()
    all_countries = sorted(df_hist['Country'].unique())
    
    current_selection_index = 0
    if st.session_state.selected_country and st.session_state.selected_country in all_countries:
        current_selection_index = all_countries.index(st.session_state.selected_country)
        
    selected_country = st.sidebar.selectbox("Select Country for Analysis:", all_countries, index=current_selection_index)
    st.session_state.selected_country = selected_country # Keep synced
    
    # 3. Data Prep for Selected Country
    c_df = df_hist[df_hist['Country'] == selected_country].sort_values('Year')
    
    if c_df.empty:
        st.warning("No data available for this country.")
    else:
        # --- FORECASTING LOGIC ---
        # Train on last 15 years for "Business as Usual" (BAU)
        train_df = c_df[c_df['Year'] >= 2008]
        if len(train_df) < 5: train_df = c_df # Fallback
        
        last_year = 2023 # Assuming dataset ends around here
        last_val = c_df['CO2 Emissions'].iloc[-1]
        
        # Future Years
        future_years = [2025, 2030, 2040, 2050]
        years_range = np.arange(last_year, 2051)
        
        # A. BAU Forecast (Linear Regression)
        if len(train_df) > 1:
            poly = np.polyfit(train_df['Year'], train_df['CO2 Emissions'], 1)
            f_func = np.poly1d(poly)
            bau_values = f_func(years_range)
            bau_values = np.maximum(bau_values, 0) # No negative emissions
        else:
            bau_values = np.array([last_val] * len(years_range)) # Flat line fallback

        # B. Net Zero Path (Linear to 0 by 2050)
        nz_slope = (0 - last_val) / (2050 - last_year)
        nz_values = [last_val + nz_slope * (y - last_year) for y in years_range]
        nz_values = np.maximum(nz_values, 0)

        # --- VISUALIZATION 1: THE 2050 COMPLIANCE CHECK ---
        col_main, col_kpi = st.columns([2.5, 1])
        
        with col_main:
            st.subheader("1. The 2050 Compliance Gap")
            fig_gap = go.Figure()
            
            # Historical
            fig_gap.add_trace(go.Scatter(
                x=c_df['Year'], y=c_df['CO2 Emissions'], mode='lines', 
                name='Historical Data', line=dict(color='#2C3E50', width=3)
            ))
            
            # BAU
            fig_gap.add_trace(go.Scatter(
                x=years_range, y=bau_values, mode='lines', 
                name='Business as Usual (Forecast)', line=dict(color='#E74C3C', dash='dash', width=2)
            ))
            
            # Net Zero
            fig_gap.add_trace(go.Scatter(
                x=years_range, y=nz_values, mode='lines', 
                name='Net Zero Path (Target)', line=dict(color='#2ECC71', dash='dot', width=2)
            ))
            
            # Fill Gap
            fig_gap.add_trace(go.Scatter(
                x=np.concatenate([years_range, years_range[::-1]]),
                y=np.concatenate([bau_values, nz_values[::-1]]),
                fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Action Gap (Reduction Needed)',
                showlegend=True
            ))
            
            # UPDATED: Fixed X-Axis Range and Scale, Better Y-Axis Label
            fig_gap.update_layout(
                title="Emissions Trajectory: Current Trend vs. Net Zero Goals",
                xaxis=dict(
                    title="Year", # Added X-axis Label
                    tickmode='array', 
                    tickvals=[2000, 2010, 2020, 2025, 2030, 2040, 2050],
                    range=[1995, 2055] # FIXED RANGE to keep ticks consistent
                ),
                yaxis_title="Annual CO2 Emissions (Million Tonnes)", # Better Y-axis Label
                legend=dict(orientation="h", y=-0.2),
                template="simple_white", height=450
            )
            st.plotly_chart(fig_gap, use_container_width=True)

        with col_kpi:
            st.subheader("Action Cards")
            
            # Calc metrics
            bau_2030 = f_func(2030) if len(train_df) > 1 else last_val
            nz_2030 = last_val + nz_slope * (2030 - last_year)
            gap_pct = ((bau_2030 - nz_2030) / bau_2030) * 100 if bau_2030 > 0 else 0
            
            # Card 1
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:8px; border-left:5px solid #2C3E50; margin-bottom:10px;">
                <h4 style="margin:0; color:#7f8c8d; font-size:14px;">2030 FORECAST (BAU)</h4>
                <p style="font-size:24px; font-weight:bold; margin:5px 0;">{bau_2030:,.1f} Mt</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 2
            color = "#E74C3C" if gap_pct > 0 else "#2ECC71"
            status = "REDUCTION NEEDED" if gap_pct > 0 else "ON TRACK"
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:8px; border-left:5px solid {color}; margin-bottom:10px;">
                <h4 style="margin:0; color:#7f8c8d; font-size:14px;">GAP TO TARGET (2030)</h4>
                <p style="font-size:24px; font-weight:bold; margin:5px 0; color:{color}">{gap_pct:.1f}%</p>
                <small style="color:{color}">{status}</small>
            </div>
            """, unsafe_allow_html=True)

            # Card 3 (Dominant Fuel)
            fuels = ['Coal CO2 Emissions', 'Oil CO2 Emissions', 'Gas CO2 Emissions']
            fuel_names = ['Coal', 'Oil', 'Gas']
            last_fuel_vals = [c_df[f].iloc[-1] for f in fuels]
            top_fuel = fuel_names[np.argmax(last_fuel_vals)]
            
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:8px; border-left:5px solid #F1C40F; margin-bottom:10px;">
                <h4 style="margin:0; color:#7f8c8d; font-size:14px;">PRIORITY SECTOR</h4>
                <p style="font-size:24px; font-weight:bold; margin:5px 0;">{top_fuel}</p>
                <small>Largest emission source</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # --- ROW 2: DETAILED INSIGHTS ---
        r2_c1, r2_c2 = st.columns(2)
        
        with r2_c1:
            # --- VISUALIZATION 2: FUTURE ENERGY COMPOSITION (Stacked Bar) ---
            st.subheader("2. Future Energy Composition")
            st.markdown("Projected breakdown of emissions by fuel source.")
            
            # Create a small forecast for each fuel
            fuel_forecasts = {}
            years_milestones = [2023, 2030, 2040, 2050]
            
            for f, name in zip(fuels, fuel_names):
                if len(train_df) > 1:
                    z = np.polyfit(train_df['Year'], train_df[f].fillna(0), 1)
                    p = np.poly1d(z)
                    vals = [max(0, p(y)) for y in years_milestones]
                else:
                    vals = [c_df[f].iloc[-1]] * 4
                fuel_forecasts[name] = vals
                
            # Plot
            fig_stack = go.Figure()
            colors = {'Coal': '#2c3e50', 'Oil': '#e67e22', 'Gas': '#f1c40f'}
            
            for name, vals in fuel_forecasts.items():
                fig_stack.add_trace(go.Bar(
                    x=years_milestones, y=vals, name=name, marker_color=colors[name]
                ))
                
            fig_stack.update_layout(
                barmode='stack',
                title="Sector-wise Emission Projections (2023-2050)",
                # UPDATED: Changed label from "Milestone Years" to "Year"
                xaxis=dict(tickmode='array', tickvals=years_milestones, title="Year"),
                # UPDATED: Matched Y-axis label with Chart 1
                yaxis_title="Annual CO2 Emissions (Million Tonnes)",
                legend=dict(orientation="h", y=-0.2),
                template="simple_white", height=400
            )
            st.plotly_chart(fig_stack, use_container_width=True)
            
        with r2_c2:
            # --- VISUALIZATION 3: DECOUPLING VELOCITY (Intensity) ---
            st.subheader("3. Decoupling Velocity")
            st.markdown("Are we getting more efficient? (CO2 per unit of GDP)")
            
            # Calculate Intensity (CO2 / GDP)
            # Use whole history for context
            if c_df['Gross Domestic Product'].count() > 5:
                # Calculate Carbon Intensity
                c_df['Intensity'] = c_df['CO2 Emissions'] / c_df['Gross Domestic Product'] * 1e9 # kg CO2 per $
                
                # Forecast GDP (Assume simplified 2% growth if no clear trend, or linear)
                fig_int = go.Figure()
                
                # UPDATED: Added Markers for clarity
                fig_int.add_trace(go.Scatter(
                    x=c_df['Year'], y=c_df['Intensity'],
                    mode='lines+markers', # Added Markers
                    marker=dict(size=6),
                    name='Carbon Intensity',
                    line=dict(color='#8E44AD', width=3)
                ))
                
                # UPDATED: Better labels and annotation
                fig_int.update_layout(
                    title="Economic Carbon Intensity Trend",
                    xaxis_title="Year",
                    yaxis_title="Emissions Efficiency (kg CO2 / $ GDP)", # Clearer Label
                    template="simple_white", height=400
                )
                
                # Added Subtitle Annotation
                fig_int.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5, y=1.1,
                    text="<i>(Lower is Better: Shows how much CO2 is emitted for every dollar of GDP generated)</i>",
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
                
                # Add annotation for latest trend
                last_5 = c_df.tail(5)
                if len(last_5) >= 2:
                    start = last_5['Intensity'].iloc[0]
                    end = last_5['Intensity'].iloc[-1]
                    if start > 0:
                        change = ((end - start) / start) * 100
                        fig_int.add_annotation(
                            x=c_df['Year'].iloc[-1], y=end,
                            text=f"Last 5Y Change: {change:.1f}%",
                            showarrow=True, arrowhead=1
                        )
                
                st.plotly_chart(fig_int, use_container_width=True)
                
            else:
                st.info("Insufficient GDP data to calculate Carbon Intensity trends.")
        
        # --- STRATEGIC ASSESSMENT TEXT ---
        st.markdown("### Strategic Assessment")
        
        trend_direction = "rising" if bau_2030 > last_val else "falling"
        
        insight_text = f"""
        Based on data from the last 15 years, **{selected_country}'s** emissions are currently **{trend_direction}**. 
        To align with a Net Zero 2050 pathway, the country needs to bridge a gap of **{gap_pct:.1f}%** by 2030. 
        The primary area for intervention is the **{top_fuel}** sector, which remains the dominant source of emissions.
        """
        st.info(insight_text)