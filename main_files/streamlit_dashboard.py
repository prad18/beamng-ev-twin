"""
🔋 EV Battery Digital Twin - Streamlit Dashboard
=================================================
Beautiful real-time dashboard for battery monitoring.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="🔋 EV Battery Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff0000;
        font-weight: bold;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Data file path
DATA_FILE = Path(__file__).parent / "live_data.json"

# Initialize session state for history
if 'soh_history' not in st.session_state:
    st.session_state.soh_history = []
if 'soc_history' not in st.session_state:
    st.session_state.soc_history = []
if 'power_history' not in st.session_state:
    st.session_state.power_history = []
if 'time_history' not in st.session_state:
    st.session_state.time_history = []


def load_data():
    """Load latest data from simulation."""
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def get_soh_status(soh):
    """Get SOH status and color."""
    if soh >= 95:
        return "EXCELLENT", "🟢", "#00ff00"
    elif soh >= 90:
        return "GOOD", "🟡", "#ffff00"
    elif soh >= 80:
        return "FAIR", "🟠", "#ff9900"
    elif soh >= 70:
        return "DEGRADED", "🔴", "#ff0000"
    else:
        return "CRITICAL", "⛔", "#990000"


def create_gauge(value, title, min_val=0, max_val=100, suffix="%", 
                 color_ranges=None, height=250):
    """Create a gauge chart."""
    if color_ranges is None:
        color_ranges = [
            [0, 0.6, "#ff0000"],
            [0.6, 0.8, "#ffaa00"],
            [0.8, 0.95, "#00aa00"],
            [0.95, 1, "#00ff00"]
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': suffix, 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': 'white'},
            'bar': {'color': "#4CAF50"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [min_val + (max_val-min_val)*r[0], 
                          min_val + (max_val-min_val)*r[1]], 
                 'color': r[2]} for r in color_ranges
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=height,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_power_gauge(power):
    """Create power gauge with regen support."""
    # Normalize to -50 to 150 range
    color = "#00ff00" if power < 0 else "#ff6600" if power > 50 else "#ffff00"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=power,
        title={'text': "⚡ Power (kW)", 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [-50, 150], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [-50, 0], 'color': '#004400'},  # Regen zone
                {'range': [0, 50], 'color': '#444400'},   # Normal
                {'range': [50, 100], 'color': '#664400'}, # High
                {'range': [100, 150], 'color': '#660000'} # Max
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_history_chart(time_data, soh_data, soc_data):
    """Create SOH/SOC history line chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_data,
        y=soh_data,
        mode='lines',
        name='SOH (%)',
        line=dict(color='#00ff00', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_data,
        y=soc_data,
        mode='lines',
        name='SOC (%)',
        line=dict(color='#00aaff', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Battery Health & Charge Over Time",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,1)',
        font={'color': 'white'},
        height=300,
        xaxis=dict(title="Simulated Days", gridcolor='rgba(100,100,100,0.3)'),
        yaxis=dict(title="Percentage (%)", range=[0, 105], gridcolor='rgba(100,100,100,0.3)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50)
    )
    
    return fig


def create_power_history(time_data, power_data):
    """Create power history area chart."""
    colors = ['#00ff00' if p < 0 else '#ff6600' for p in power_data]
    
    fig = go.Figure()
    
    # Positive power (discharge)
    pos_power = [max(0, p) for p in power_data]
    fig.add_trace(go.Scatter(
        x=time_data, y=pos_power,
        fill='tozeroy', name='Discharge',
        line=dict(color='#ff6600'), fillcolor='rgba(255,102,0,0.3)'
    ))
    
    # Negative power (regen)
    neg_power = [min(0, p) for p in power_data]
    fig.add_trace(go.Scatter(
        x=time_data, y=neg_power,
        fill='tozeroy', name='Regen',
        line=dict(color='#00ff00'), fillcolor='rgba(0,255,0,0.3)'
    ))
    
    fig.update_layout(
        title="Power Flow (Discharge ↑ / Regen ↓)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,1)',
        font={'color': 'white'},
        height=200,
        xaxis=dict(title="Simulated Days", gridcolor='rgba(100,100,100,0.3)'),
        yaxis=dict(title="Power (kW)", gridcolor='rgba(100,100,100,0.3)'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=50)
    )
    
    return fig


def format_time(days):
    """Format days as readable time."""
    if days < 1:
        return f"{days * 24:.1f} hours"
    elif days < 30:
        return f"{days:.0f} days"
    elif days < 365:
        return f"{days / 30:.1f} months"
    else:
        return f"{days / 365:.2f} years"


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔋 EV Battery Digital Twin Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.5, 5.0, 1.0, 0.5)
        st.markdown("---")
        
        st.header("📊 About")
        st.info("""
        This dashboard displays real-time battery data from the BeamNG EV simulation.
        
        **Run the simulation:**
        ```
        python demo_simulation.py
        ```
        
        **Features:**
        - Real-time SOC/SOH tracking
        - Power consumption monitoring
        - Degradation visualization
        - Time acceleration (10,000x)
        """)
        
        if st.button("🗑️ Clear History"):
            st.session_state.soh_history = []
            st.session_state.soc_history = []
            st.session_state.power_history = []
            st.session_state.time_history = []
            st.rerun()
    
    # Load data
    data = load_data()
    
    if data is None:
        st.warning("⏳ Waiting for simulation data... Make sure `demo_simulation.py` is running!")
        
        # Show placeholder
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔋 SOC", "-- %")
        with col2:
            st.metric("💚 SOH", "-- %")
        with col3:
            st.metric("⚡ Power", "-- kW")
        
        time.sleep(refresh_rate)
        st.rerun()
        return
    
    # Extract data
    soc = data.get('soc', 0)
    soh = data.get('soh', 100)
    temp = data.get('temperature', 25)
    power = data.get('power_kw', 0)
    voltage = data.get('voltage', 370)
    current = data.get('current', 0)
    sim_days = data.get('simulated_days', 0)
    distance = data.get('total_distance_km', 0)
    cycles = data.get('cycle_count', 0)
    elapsed = data.get('elapsed_seconds', 0)
    running = data.get('simulation_running', False)
    demo_mode = data.get('demo_mode', False)
    vehicle = data.get('vehicle_model', 'Unknown')
    is_regen = data.get('is_regen', False)
    
    # Update history (limit to 500 points)
    if len(st.session_state.time_history) == 0 or \
       sim_days != st.session_state.time_history[-1]:
        st.session_state.time_history.append(sim_days)
        st.session_state.soh_history.append(soh)
        st.session_state.soc_history.append(soc)
        st.session_state.power_history.append(power)
        
        # Limit history
        max_points = 500
        if len(st.session_state.time_history) > max_points:
            st.session_state.time_history = st.session_state.time_history[-max_points:]
            st.session_state.soh_history = st.session_state.soh_history[-max_points:]
            st.session_state.soc_history = st.session_state.soc_history[-max_points:]
            st.session_state.power_history = st.session_state.power_history[-max_points:]
    
    # Status bar
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        status_icon = "🟢" if running else "🔴"
        status_text = "RUNNING" if running else "STOPPED"
        st.markdown(f"### {status_icon} Status: **{status_text}**")
    with status_col2:
        st.markdown(f"### 🚗 Vehicle: **{vehicle}**")
    with status_col3:
        mode_text = "DEMO (10,000x)" if demo_mode else "REALISTIC"
        st.markdown(f"### ⚡ Mode: **{mode_text}**")
    with status_col4:
        soh_status, soh_emoji, soh_color = get_soh_status(soh)
        st.markdown(f"### {soh_emoji} Health: **{soh_status}**")
    
    st.markdown("---")
    
    # Main gauges row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        soc_gauge = create_gauge(soc, "🔋 State of Charge", 0, 100, "%",
                                 [[0, 0.2, "#ff0000"], [0.2, 0.5, "#ffaa00"],
                                  [0.5, 0.8, "#00aa00"], [0.8, 1, "#00ff00"]])
        st.plotly_chart(soc_gauge, use_container_width=True)
    
    with col2:
        soh_gauge = create_gauge(soh, "💚 State of Health", 60, 100, "%",
                                 [[0, 0.5, "#ff0000"], [0.5, 0.75, "#ffaa00"],
                                  [0.75, 0.9, "#00aa00"], [0.9, 1, "#00ff00"]])
        st.plotly_chart(soh_gauge, use_container_width=True)
    
    with col3:
        power_gauge = create_power_gauge(power)
        st.plotly_chart(power_gauge, use_container_width=True)
    
    with col4:
        temp_gauge = create_gauge(temp, "🌡️ Temperature", 20, 60, "°C",
                                  [[0, 0.3, "#00aaff"], [0.3, 0.6, "#00ff00"],
                                   [0.6, 0.8, "#ffaa00"], [0.8, 1, "#ff0000"]])
        st.plotly_chart(temp_gauge, use_container_width=True)
    
    st.markdown("---")
    
    # Time simulation progress
    st.subheader("⏱️ Simulated Time Progress")
    
    progress = min(1.0, sim_days / 730)
    st.progress(progress)
    
    time_col1, time_col2, time_col3 = st.columns(3)
    with time_col1:
        st.metric("📅 Simulated Days", f"{sim_days:.0f} / 730")
    with time_col2:
        st.metric("📆 Simulated Time", format_time(sim_days))
    with time_col3:
        st.metric("⏱️ Real Time Elapsed", f"{elapsed:.0f} sec")
    
    st.markdown("---")
    
    # History charts
    if len(st.session_state.time_history) > 1:
        st.subheader("📈 Real-Time Monitoring")
        
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            history_chart = create_history_chart(
                st.session_state.time_history,
                st.session_state.soh_history,
                st.session_state.soc_history
            )
            st.plotly_chart(history_chart, use_container_width=True)
        
        with chart_col2:
            # Stats
            st.markdown("### 📊 Statistics")
            
            soh_drop = 100 - soh
            if sim_days > 0:
                daily_degradation = soh_drop / sim_days
                predicted_2yr = 100 - (daily_degradation * 730)
                years_to_80 = (soh - 80) / daily_degradation / 365 if daily_degradation > 0 else float('inf')
            else:
                predicted_2yr = 100
                years_to_80 = float('inf')
            
            st.metric("SOH Lost", f"{soh_drop:.3f}%")
            st.metric("Predicted 2-Year SOH", f"{max(0, predicted_2yr):.1f}%")
            st.metric("Years to 80% SOH", f"{min(20, years_to_80):.1f}" if years_to_80 < 100 else "20+")
        
        # Power history
        power_chart = create_power_history(
            st.session_state.time_history,
            st.session_state.power_history
        )
        st.plotly_chart(power_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Driving statistics
    st.subheader("🚗 Driving Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("🛣️ Distance", f"{distance:.1f} km")
    with stat_col2:
        st.metric("🔄 Charge Cycles", f"{cycles:.1f}")
    with stat_col3:
        st.metric("⚡ Voltage", f"{voltage:.1f} V")
    with stat_col4:
        current_direction = "↓ Discharge" if current > 0 else "↑ Charge" if current < 0 else "Idle"
        st.metric("🔌 Current", f"{abs(current):.1f} A {current_direction}")
    
    # Regen indicator
    if is_regen:
        st.success("🔋 REGENERATIVE BRAKING ACTIVE - Energy being recovered!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"Last Update: {data.get('timestamp', 'N/A')[:19]} | "
        f"Refresh Rate: {refresh_rate}s</div>",
        unsafe_allow_html=True
    )
    
    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
