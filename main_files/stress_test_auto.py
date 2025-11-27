"""
EV Battery Stress Test - Automated Aggressive Driving
======================================================
Runs a timed stress test with aggressive driving patterns.

Usage:
    python stress_test_auto.py
    python stress_test_auto.py --duration 5   # 5 minute test
    python stress_test_auto.py --duration 15  # 15 minute test (default)
"""

import sys
import os
import time
import json
import math
import random
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import requests

# BeamNG imports
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, PowertrainSensor

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = Path(__file__).parent / "config.yaml"
DATA_FILE = Path(__file__).parent / "live_data.json"
REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)

CFG = load_config()

# PyBaMM API
_twin_url = CFG.get('twin_url', 'http://127.0.0.1:8008/step')
TWIN_API_URL = _twin_url.replace('/step', '').rstrip('/')

# Battery specs
BATTERY_CAPACITY_KWH = 81.4
BATTERY_CAPACITY_AH = 220
NOMINAL_VOLTAGE = 370
TIME_MULTIPLIER = 10000

# ============================================================================
# AGGRESSIVE DRIVER AI
# ============================================================================

class AggressiveDriver:
    """AI that drives aggressively to stress the battery."""
    
    def __init__(self):
        self.pattern_time = 0
        self.pattern_duration = random.uniform(8, 20)
        self.patterns = [
            "full_throttle",
            "hard_braking",
            "throttle_spam",
            "regen_coast",
            "acceleration_test"
        ]
        self.current_pattern = random.choice(self.patterns)
        
    def get_inputs(self, vehicle_speed_kmh, elapsed):
        """Get throttle/brake/steering inputs."""
        self.pattern_time += 0.1
        
        # Switch patterns periodically
        if self.pattern_time > self.pattern_duration:
            self.current_pattern = random.choice(self.patterns)
            self.pattern_duration = random.uniform(8, 20)
            self.pattern_time = 0
        
        if self.current_pattern == "full_throttle":
            return 1.0, 0.0, random.uniform(-0.1, 0.1)
            
        elif self.current_pattern == "hard_braking":
            if vehicle_speed_kmh > 20:
                return 0.0, 1.0, 0.0
            else:
                return 1.0, 0.0, 0.0
                
        elif self.current_pattern == "throttle_spam":
            if random.random() > 0.4:
                return 1.0, 0.0, random.uniform(-0.15, 0.15)
            else:
                return 0.0, 0.5, 0.0
                
        elif self.current_pattern == "regen_coast":
            if vehicle_speed_kmh > 30:
                return 0.0, 0.3, random.uniform(-0.1, 0.1)
            else:
                return 0.8, 0.0, 0.0
                
        else:  # acceleration_test
            return 1.0, 0.0, 0.0

# ============================================================================
# PYBAMM CLIENT
# ============================================================================

class PyBaMMClient:
    """Client for PyBaMM API."""
    
    def __init__(self):
        self.connected = False
        self.model_type = "Fallback"
        self.step_count = 0
        self._check_connection()
        
    def _check_connection(self):
        try:
            r = requests.get(f"{TWIN_API_URL}/", timeout=2.0)
            if r.status_code == 200:
                info = r.json()
                self.connected = True
                self.model_type = info.get('model', 'Unknown')
                print(f"✅ PyBaMM API: {self.model_type}")
                return
        except:
            pass
        print("⚠️ PyBaMM API not available - using fallback model")
        
    def step(self, current_a, soc, temp_c, dt_s, accel_dt=None):
        if not self.connected:
            return None
        try:
            payload = {
                "pack_current_A": float(current_a),
                "soc": float(soc),
                "amb_temp_C": float(temp_c),
                "dt_s": float(min(dt_s, 60))
            }
            if accel_dt:
                payload["accelerated_dt_s"] = float(accel_dt)
            r = requests.post(f"{TWIN_API_URL}/step", json=payload, timeout=2.0)
            if r.status_code == 200:
                self.step_count += 1
                return r.json()
        except:
            pass
        return None
    
    def reset(self):
        try:
            requests.post(f"{TWIN_API_URL}/reset", params={"soc": 0.8, "soh": 1.0}, timeout=2.0)
        except:
            pass

# ============================================================================
# BATTERY TRACKER
# ============================================================================

class BatteryTracker:
    """Track battery state during stress test."""
    
    def __init__(self, pybamm):
        self.pybamm = pybamm
        self.soc = 0.80
        self.soh = 1.0
        self.start_soh = 1.0
        self.temperature = 25.0
        self.voltage = NOMINAL_VOLTAGE
        self.current = 0.0
        self.power_kw = 0.0
        
        self.start_time = time.time()
        self.simulated_seconds = 0
        self.total_energy_kwh = 0
        self.total_distance_km = 0
        self.cycle_count = 0
        self.peak_power = 0
        self.peak_current = 0
        self.max_temp = 25
        self.min_soc = 1.0
        
        self.history = []
        
    def update(self, motor_torque, motor_speed, vehicle_speed, dt):
        """Update battery from motor data."""
        self.power_kw = (motor_torque * motor_speed) / 1000
        self.current = (self.power_kw * 1000) / max(self.voltage, 300)
        
        # Track peaks
        self.peak_power = max(self.peak_power, abs(self.power_kw))
        self.peak_current = max(self.peak_current, abs(self.current))
        
        # Update SOC
        energy = self.power_kw * (dt / 3600)
        self.soc = max(0.05, min(0.95, self.soc - energy / BATTERY_CAPACITY_KWH))
        self.min_soc = min(self.min_soc, self.soc)
        
        self.total_energy_kwh += abs(energy)
        self.total_distance_km += (vehicle_speed * dt) / 1000
        
        # Accelerated time
        accel_dt = dt * TIME_MULTIPLIER
        self.simulated_seconds += accel_dt
        
        # PyBaMM step
        result = self.pybamm.step(self.current, self.soc, self.temperature, dt, accel_dt)
        
        if result:
            self.soh = result.get('soh', self.soh)
            self.temperature = result.get('pack_temp_C', self.temperature)
            self.voltage = result.get('pack_voltage_V', self.voltage)
            self.cycle_count = result.get('cycle_count', self.cycle_count)
        else:
            # Fallback degradation
            c_rate = abs(self.current) / BATTERY_CAPACITY_AH
            deg = 2e-8 * (c_rate ** 1.5) * accel_dt + 1e-9 * accel_dt
            self.soh = max(0.6, self.soh - deg)
            
        self.max_temp = max(self.max_temp, self.temperature)
        
        # Record history periodically
        elapsed = time.time() - self.start_time
        if len(self.history) == 0 or elapsed - (self.history[-1]['t'] if self.history else 0) > 30:
            self.history.append({
                't': elapsed,
                'days': self.simulated_seconds / 86400,
                'soc': self.soc,
                'soh': self.soh,
                'temp': self.temperature,
                'power': self.power_kw
            })
            
    def get_state(self):
        elapsed = time.time() - self.start_time
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "simulated_days": self.simulated_seconds / 86400,
            "simulated_years": self.simulated_seconds / 86400 / 365,
            "soc": self.soc * 100,
            "soh": self.soh * 100,
            "temperature": self.temperature,
            "voltage": self.voltage,
            "current": self.current,
            "power_kw": self.power_kw,
            "total_distance_km": self.total_distance_km,
            "cycle_count": self.cycle_count,
            "energy_throughput_kwh": self.total_energy_kwh,
            "demo_mode": True,
            "time_multiplier": TIME_MULTIPLIER,
            "model_type": self.pybamm.model_type,
            "pybamm_connected": self.pybamm.connected,
            "peak_power_kw": self.peak_power,
            "peak_current_a": self.peak_current,
            "stress_test": True
        }

# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(battery, pybamm, duration_mins):
    """Generate readable report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elapsed = time.time() - battery.start_time
    sim_days = battery.simulated_seconds / 86400
    soh_loss = (battery.start_soh - battery.soh) * 100
    
    report = f"""
================================================================================
              EV BATTERY STRESS TEST REPORT
================================================================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SIMULATION SUMMARY
------------------
  Real Duration:       {elapsed/60:.1f} minutes
  Simulated Time:      {sim_days:.1f} days ({sim_days/365:.2f} years)
  Time Acceleration:   {TIME_MULTIPLIER:,}x

BATTERY RESULTS
---------------
  Starting SOH:        {battery.start_soh*100:.1f}%
  Final SOH:           {battery.soh*100:.2f}%
  Degradation:         {soh_loss:.3f}%
  Equivalent Cycles:   {battery.cycle_count:.1f}
  
  Original Capacity:   {BATTERY_CAPACITY_KWH:.1f} kWh
  Remaining Capacity:  {BATTERY_CAPACITY_KWH * battery.soh:.1f} kWh
  Capacity Lost:       {BATTERY_CAPACITY_KWH * (1 - battery.soh):.2f} kWh

STRESS METRICS
--------------
  Distance Driven:     {battery.total_distance_km:.1f} km
  Energy Used:         {battery.total_energy_kwh:.1f} kWh
  Peak Power:          {battery.peak_power:.1f} kW
  Peak Current:        {battery.peak_current:.1f} A
  Max Temperature:     {battery.max_temp:.1f} C
  Min SOC Reached:     {battery.min_soc*100:.1f}%

MODEL INFO
----------
  Physics Model:       {pybamm.model_type}
  Simulation Steps:    {pybamm.step_count:,}
  Battery Chemistry:   LiFePO4 (LFP)
  Battery Pack:        {BATTERY_CAPACITY_KWH} kWh / {NOMINAL_VOLTAGE}V

HISTORY
-------
  Time(min) | Sim Days | SOC%  | SOH%   | Temp C | Power kW
  -------------------------------------------------------
"""
    for h in battery.history:
        report += f"  {h['t']/60:8.1f} | {h['days']:8.1f} | {h['soc']*100:5.1f} | {h['soh']*100:6.2f} | {h['temp']:6.1f} | {h['power']:8.1f}\n"
    
    report += """
================================================================================
                         END OF REPORT
================================================================================
"""
    
    # Save report
    report_file = REPORT_DIR / f"stress_test_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON
    json_file = REPORT_DIR / f"stress_test_{timestamp}.json"
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": elapsed / 60,
        "simulated_days": sim_days,
        "start_soh": battery.start_soh,
        "final_soh": battery.soh,
        "soh_loss_percent": soh_loss,
        "peak_power_kw": battery.peak_power,
        "peak_current_a": battery.peak_current,
        "max_temp_c": battery.max_temp,
        "history": battery.history
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n📄 Report: {report_file}")
    print(f"📊 Data:   {json_file}")
    
    return report

# ============================================================================
# MAIN STRESS TEST
# ============================================================================

def run_stress_test(duration_mins=15, vehicle_model="vivace"):
    """Run the stress test."""
    
    print("""
================================================================================
         EV BATTERY STRESS TEST - AGGRESSIVE DRIVING
================================================================================
    """)
    
    # Initialize PyBaMM
    pybamm = PyBaMMClient()
    pybamm.reset()
    
    # Initialize battery tracker
    battery = BatteryTracker(pybamm)
    
    # Initialize driver AI
    driver = AggressiveDriver()
    
    # Connect to BeamNG
    print("\n🎮 Connecting to BeamNG.tech...")
    
    try:
        beamng = BeamNGpy('localhost', 64256, home=CFG['beamng_home'])
        beamng.open()
        print("✅ Connected to BeamNG")
        
        # Clean up any existing scenario first
        try:
            beamng.restart_scenario()
        except:
            pass
        
        # Create vehicle with unique name
        vehicle = Vehicle('ev_test_car', model=vehicle_model)
        print(f"🚗 Using vehicle: {vehicle_model}")
        
        # Create scenario on a different spawn point
        scenario = Scenario('smallgrid', 'ev_stress_test')
        scenario.add_vehicle(vehicle, pos=(0, 0, 0.5))  # Slightly elevated to avoid ground clipping
        scenario.make(beamng)
        
        beamng.load_scenario(scenario)
        beamng.start_scenario()
        beamng.switch_vehicle(vehicle)
        print("✅ Scenario loaded")
        
        # Attach sensors
        vehicle.sensors.attach('electrics', Electrics())
        
        # PowertrainSensor
        try:
            powertrain = PowertrainSensor('powertrain', beamng, vehicle)
        except:
            powertrain = None
            print("⚠️ PowertrainSensor not available")
        
        # Stabilize
        for _ in range(10):
            beamng.step(1)
            time.sleep(0.05)
        
        print(f"\n⏱️ Starting {duration_mins}-minute stress test...")
        print("   Press Ctrl+C to stop early\n")
        
        # Main loop
        start_time = time.time()
        end_time = start_time + (duration_mins * 60)
        last_update = start_time
        last_print = start_time
        
        while time.time() < end_time:
            now = time.time()
            dt = now - last_update
            last_update = now
            
            # Get sensor data
            vehicle.sensors.poll()
            electrics = vehicle.sensors._sensors.get('electrics', {})
            e_data = electrics.data if hasattr(electrics, 'data') else {}
            
            vehicle_speed = e_data.get('wheelspeed', 0) or 0
            
            # Get motor data from powertrain
            motor_torque = 0
            motor_speed = 0
            if powertrain:
                try:
                    powertrain.poll()
                    for name, data in powertrain.data.items():
                        if 'motor' in name.lower():
                            motor_torque = data.get('outputTorque', 0) or 0
                            motor_speed = data.get('outputAV', 0) or 0
                            break
                except:
                    pass
            
            # Simulate motor data if not available
            if motor_torque == 0 and motor_speed == 0:
                motor_speed = vehicle_speed * 10  # Rough approximation
                motor_torque = random.uniform(50, 200) if vehicle_speed > 1 else 0
            
            # Get driver inputs
            speed_kmh = vehicle_speed * 3.6
            throttle, brake, steering = driver.get_inputs(speed_kmh, now - start_time)
            
            # Apply controls
            vehicle.control(throttle=throttle, brake=brake, steering=steering)
            
            # Step simulation
            beamng.step(3)
            
            # Update battery
            battery.update(motor_torque, motor_speed, vehicle_speed, dt)
            
            # Save state for dashboard
            state = battery.get_state()
            remaining = end_time - now
            state['remaining_seconds'] = remaining
            state['progress_percent'] = ((now - start_time) / (duration_mins * 60)) * 100
            state['driver_pattern'] = driver.current_pattern
            
            with open(DATA_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Print progress
            if now - last_print > 30:
                sim_days = battery.simulated_seconds / 86400
                print(f"  [{(now-start_time)/60:5.1f}m] SOH: {battery.soh*100:.2f}% | "
                      f"SOC: {battery.soc*100:.0f}% | Sim: {sim_days:.0f} days | {driver.current_pattern}")
                last_print = now
            
            time.sleep(0.05)
        
        print("\n✅ Stress test completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test stopped by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Generate report
        print("\n📊 Generating report...")
        report = generate_report(battery, pybamm, duration_mins)
        
        # Print summary
        print("\n" + "="*50)
        print("                SUMMARY")
        print("="*50)
        print(f"  Duration:      {(time.time()-battery.start_time)/60:.1f} min")
        print(f"  Simulated:     {battery.simulated_seconds/86400:.0f} days")
        print(f"  SOH Change:    {battery.start_soh*100:.1f}% -> {battery.soh*100:.2f}%")
        print(f"  Degradation:   {(battery.start_soh-battery.soh)*100:.3f}%")
        print("="*50)
        
        # Close BeamNG
        try:
            beamng.close()
        except:
            pass

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Battery Stress Test")
    parser.add_argument('--duration', type=int, default=15, help='Duration in minutes (default: 15)')
    parser.add_argument('--vehicle', type=str, default='sv1ev3', help='Vehicle model (default: sv1ev3 = Kia EV3)')
    
    args = parser.parse_args()
    
    run_stress_test(duration_mins=args.duration, vehicle_model=args.vehicle)
