"""
🔋 EV Battery Digital Twin - Live Dashboard
============================================
Real-time terminal dashboard that displays battery status.
Run this alongside demo_simulation.py for the split-screen effect.

Usage:
  Terminal 1: python demo_simulation.py
  Terminal 2: python dashboard.py
"""

import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Data file shared with simulation
DATA_FILE = Path(__file__).parent / "live_data.json"


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def create_bar(value, max_value, width=20, fill_char="█", empty_char="░"):
    """Create a progress bar string."""
    if max_value == 0:
        return empty_char * width
    filled = int((value / max_value) * width)
    filled = max(0, min(width, filled))
    return fill_char * filled + empty_char * (width - filled)


def get_soh_color(soh):
    """Get status text based on SOH."""
    if soh >= 95:
        return "EXCELLENT", "💚"
    elif soh >= 90:
        return "GOOD", "💛"
    elif soh >= 80:
        return "FAIR", "🟠"
    elif soh >= 70:
        return "DEGRADED", "🔴"
    else:
        return "CRITICAL", "⛔"


def get_power_indicator(power_kw):
    """Get power status indicator."""
    if power_kw < -5:
        return "⚡ REGEN", "🔋↑"
    elif power_kw < 0:
        return "Regen", "🔋↑"
    elif power_kw < 10:
        return "Idle/Low", "🔋"
    elif power_kw < 50:
        return "Moderate", "🔋↓"
    elif power_kw < 100:
        return "High", "⚡↓"
    else:
        return "MAX POWER", "🔥↓"


def format_time(days):
    """Format simulated days as readable time."""
    if days < 1:
        return f"{days * 24:.1f} hours"
    elif days < 30:
        return f"{days:.1f} days"
    elif days < 365:
        months = days / 30
        return f"{months:.1f} months"
    else:
        years = days / 365
        return f"{years:.2f} years"


def load_data():
    """Load latest data from simulation."""
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def render_dashboard(data):
    """Render the dashboard display."""
    if data is None:
        return """
╔═══════════════════════════════════════════════════════════════════╗
║  🔋 EV BATTERY DIGITAL TWIN - DASHBOARD                           ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║           ⏳ Waiting for simulation data...                       ║
║                                                                    ║
║           Run demo_simulation.py in another terminal              ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    
    # Extract values
    soc = data.get('soc', 0)
    soh = data.get('soh', 100)
    temp = data.get('temperature', 25)
    power = data.get('power_kw', 0)
    voltage = data.get('voltage', 370)
    current = data.get('current', 0)
    
    sim_days = data.get('simulated_days', 0)
    sim_years = data.get('simulated_years', 0)
    distance = data.get('total_distance_km', 0)
    cycles = data.get('cycle_count', 0)
    elapsed = data.get('elapsed_seconds', 0)
    
    is_regen = data.get('is_regen', False)
    demo_mode = data.get('demo_mode', False)
    vehicle = data.get('vehicle_model', 'Unknown')
    running = data.get('simulation_running', False)
    
    # Create visual elements
    soc_bar = create_bar(soc, 100, 20)
    soh_bar = create_bar(soh, 100, 20)
    temp_bar = create_bar(temp - 20, 40, 20)  # 20-60°C range
    
    soh_status, soh_emoji = get_soh_color(soh)
    power_status, power_emoji = get_power_indicator(power)
    
    # Progress through 2 years
    progress_pct = min(100, (sim_days / 730) * 100)
    progress_bar = create_bar(progress_pct, 100, 40)
    
    # Status indicator
    status = "🟢 RUNNING" if running else "🔴 STOPPED"
    mode = "DEMO (10,000x)" if demo_mode else "REALISTIC"
    
    # Power graph (simple ASCII)
    power_normalized = min(150, max(-50, power))
    power_pos = int((power_normalized + 50) / 200 * 40)
    power_graph = "─" * power_pos + "│" + "─" * (40 - power_pos)
    
    dashboard = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  🔋 EV BATTERY DIGITAL TWIN - LIVE DASHBOARD              {status:14s} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  🚗 Vehicle: {vehicle:20s}              ⚡ Mode: {mode:14s}    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌────────────────────────┐  ┌────────────────────────┐  ┌──────────────┐ ║
║  │  🔋 SOC (Charge)       │  │  💚 SOH (Health)       │  │  🌡️ Temp     │ ║
║  │  {soc:5.1f}%              │  │  {soh:5.2f}%             │  │  {temp:4.1f}°C    │ ║
║  │  {soc_bar} │  │  {soh_bar} │  │  {temp_bar:.10s} │ ║
║  └────────────────────────┘  └────────────────────────┘  └──────────────┘ ║
║                                                                            ║
║  ⚡ POWER: {power:+7.1f} kW   {power_emoji} {power_status:12s}                            ║
║  ┌────────────────────────────────────────────────────────────────────┐   ║
║  │ -50kW {power_graph} +150kW │   ║
║  │       REGEN ◄────────────────────────────────► DISCHARGE          │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
║  📊 BATTERY STATUS: {soh_emoji} {soh_status:12s}                                       ║
║  ═══════════════════════════════════════════════════════════════════════  ║
║  │ Voltage: {voltage:6.1f}V │ Current: {current:+7.1f}A │ Resistance: Growing │       ║
║                                                                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  ⏱️ SIMULATED TIME                                                        ║
║  ══════════════════════════════════════════════════════════════════════   ║
║                                                                            ║
║  Day {sim_days:6.0f} of 730  ({format_time(sim_days):>15s})                             ║
║  {progress_bar}  {progress_pct:5.1f}%    ║
║  ├─────────────────────────────────────────────────────────────────────┤  ║
║  Year 0                    Year 1                    Year 2              ║
║                                                                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  📈 DRIVING STATISTICS                                                    ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           ║
║  │ Distance        │  │ Charge Cycles   │  │ Real Time       │           ║
║  │ {distance:8.1f} km    │  │ {cycles:8.1f}       │  │ {elapsed:8.1f} sec   │           ║
║  └─────────────────┘  └─────────────────┘  └─────────────────┘           ║
║                                                                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  🔮 PREDICTION (at current rate)                                          ║
║  ───────────────────────────────────────────────────────────────────────  ║
║  │ After 2 years: ~{100 - (100 - soh) * (730 / max(1, sim_days)):5.1f}% SOH                                          │  ║
║  │ Battery end-of-life (80% SOH): ~{max(0, 730 * (soh - 80) / max(0.01, 100 - soh) / 365):4.1f} years                              │  ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
    Last Update: {data.get('timestamp', 'N/A')[:19]}
"""
    return dashboard


def main():
    print("🔋 EV Battery Digital Twin - Live Dashboard")
    print("=" * 50)
    print("Connecting to simulation data...")
    print("(Make sure demo_simulation.py is running)")
    print("\nPress Ctrl+C to exit\n")
    
    time.sleep(1)
    
    try:
        while True:
            clear_screen()
            
            # Load latest data
            data = load_data()
            
            # Render dashboard
            dashboard = render_dashboard(data)
            print(dashboard)
            
            # Refresh rate
            time.sleep(0.25)  # 4 updates per second
            
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed")


if __name__ == "__main__":
    main()
