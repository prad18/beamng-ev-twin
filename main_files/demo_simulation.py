"""
🔋 EV Battery Digital Twin - Live Demo Simulation
==================================================
This script connects to BeamNG and runs a live battery simulation
with accelerated degradation for demo purposes.

Run this alongside dashboard.py for the split-screen demo effect.
"""

import time
import json
import math
import os
import yaml
from datetime import datetime
from pathlib import Path

# BeamNG imports
try:
    from beamngpy import BeamNGpy, Vehicle, Scenario
    from beamngpy.sensors import Electrics, PowertrainSensor
    BEAMNG_AVAILABLE = True
except ImportError:
    print("❌ beamngpy not installed. Run: pip install beamngpy")
    BEAMNG_AVAILABLE = False

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
if CONFIG_PATH.exists():
    CFG = yaml.safe_load(open(CONFIG_PATH, 'r'))
else:
    # Fallback to beamng_client config
    CFG = yaml.safe_load(open(Path(__file__).parent.parent / "beamng_client" / "config.yaml", 'r'))

# Shared data file for dashboard communication
DATA_FILE = Path(__file__).parent / "live_data.json"


class BatteryState:
    """Battery state with realistic physics and accelerated degradation."""
    
    def __init__(self, capacity_ah=220, voltage=370, demo_mode=True):
        # Battery specifications
        self.capacity_ah = capacity_ah  # Amp-hours
        self.nominal_voltage = voltage  # Volts
        self.energy_kwh = (capacity_ah * voltage) / 1000  # Total energy
        
        # Current state
        self.soc = 0.85  # State of Charge (85% starting)
        self.soh = 1.0   # State of Health (100% = new battery)
        self.temperature = 25.0  # Cell temperature in °C
        self.voltage = voltage
        self.current = 0.0  # Amps (positive = discharge, negative = charge)
        self.power_kw = 0.0
        
        # Demo mode acceleration
        self.demo_mode = demo_mode
        self.time_multiplier = 10000 if demo_mode else 1  # 10,000x for demo
        
        # Degradation tracking
        self.total_energy_throughput = 0.0  # kWh cycled
        self.simulated_days = 0.0
        self.cycle_count = 0
        self.total_distance_km = 0.0
        
        # Internal resistance (grows with degradation)
        self.internal_resistance = 0.05  # Ohms (new battery)
        
        # Tracking
        self.start_time = time.time()
        self.last_update = time.time()
        
    def update(self, motor_torque, motor_speed, dt, vehicle_speed_mps=0):
        """Update battery state based on motor data."""
        now = time.time()
        real_dt = now - self.last_update
        self.last_update = now
        
        # Calculate power from motor
        self.power_kw = (motor_torque * motor_speed) / 1000
        
        # Calculate current
        self.current = (self.power_kw * 1000) / self.voltage
        
        # C-rate (how fast we're charging/discharging relative to capacity)
        c_rate = abs(self.current) / self.capacity_ah
        
        # Update SOC (realistic, not accelerated)
        energy_delta = self.power_kw * (real_dt / 3600)  # kWh
        soc_delta = energy_delta / self.energy_kwh
        self.soc = max(0.0, min(1.0, self.soc - soc_delta))
        
        # Track energy throughput
        self.total_energy_throughput += abs(energy_delta)
        
        # Update temperature (simplified thermal model)
        heat_generated = (self.current ** 2) * self.internal_resistance * real_dt
        cooling = 0.1 * (self.temperature - 25) * real_dt  # Passive cooling
        self.temperature += (heat_generated * 0.01) - cooling
        self.temperature = max(20, min(60, self.temperature))  # Clamp
        
        # Update voltage based on SOC and load
        ocv = self.nominal_voltage * (0.9 + 0.2 * self.soc)  # Open circuit voltage
        self.voltage = ocv - (self.current * self.internal_resistance)
        
        # Track distance
        if vehicle_speed_mps > 0:
            self.total_distance_km += (vehicle_speed_mps * real_dt) / 1000
        
        # === ACCELERATED DEGRADATION (for demo) ===
        if self.demo_mode:
            accelerated_dt = real_dt * self.time_multiplier
        else:
            accelerated_dt = real_dt
        
        # Cycle degradation (based on power usage)
        cycle_stress = c_rate ** 1.5 if c_rate > 0 else 0
        cycle_degradation = 2e-8 * cycle_stress * accelerated_dt
        
        # Temperature degradation (Arrhenius-like)
        temp_factor = math.exp((self.temperature - 25) / 15)
        thermal_degradation = 1e-9 * temp_factor * accelerated_dt
        
        # Calendar aging (always happens)
        soc_stress = 1.5 if self.soc > 0.8 else (1.3 if self.soc < 0.2 else 1.0)
        calendar_degradation = 5e-10 * soc_stress * accelerated_dt
        
        # Total degradation
        total_degradation = cycle_degradation + thermal_degradation + calendar_degradation
        self.soh = max(0.6, self.soh - total_degradation)  # Min 60% SOH
        
        # Update internal resistance (grows as battery degrades)
        self.internal_resistance = 0.05 / (self.soh ** 2)
        
        # Track simulated time
        self.simulated_days += accelerated_dt / 86400  # Convert to days
        
        # Track cycles (rough estimate: 1 cycle = 2 * capacity discharged)
        self.cycle_count = self.total_energy_throughput / (2 * self.energy_kwh)
        
    def get_state_dict(self):
        """Return current state as dictionary for dashboard."""
        elapsed_real = time.time() - self.start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_real,
            "simulated_days": self.simulated_days,
            "simulated_years": self.simulated_days / 365,
            
            # Battery state
            "soc": self.soc * 100,  # As percentage
            "soh": self.soh * 100,  # As percentage
            "temperature": self.temperature,
            "voltage": self.voltage,
            "current": self.current,
            "power_kw": self.power_kw,
            
            # Driving stats
            "total_distance_km": self.total_distance_km,
            "cycle_count": self.cycle_count,
            "energy_throughput_kwh": self.total_energy_throughput,
            
            # Status flags
            "is_charging": self.power_kw < 0,
            "is_regen": self.power_kw < -1,
            "demo_mode": self.demo_mode,
            "time_multiplier": self.time_multiplier
        }


class LiveDemoSimulation:
    """Main simulation controller for live demos."""
    
    def __init__(self, vehicle_model="sv1ev3", demo_mode=True):
        self.vehicle_model = vehicle_model
        self.demo_mode = demo_mode
        self.beamng = None
        self.vehicle = None
        self.powertrain_sensor = None
        self.battery = BatteryState(demo_mode=demo_mode)
        self.running = False
        
    def connect(self):
        """Connect to BeamNG and set up vehicle."""
        print("🔌 Connecting to BeamNG.tech...")
        
        self.beamng = BeamNGpy('localhost', 64256, home=CFG['beamng_home'])
        self.beamng.open()
        print("✅ Connected to BeamNG")
        
        # Create vehicle
        self.vehicle = Vehicle('ego', model=self.vehicle_model)
        print(f"🚗 Created vehicle: {self.vehicle_model}")
        
        # Create scenario
        scenario = Scenario('smallgrid', 'battery_demo')
        scenario.add_vehicle(self.vehicle, pos=(0, 0, 0))
        scenario.make(self.beamng)
        
        self.beamng.load_scenario(scenario)
        self.beamng.start_scenario()
        self.beamng.switch_vehicle(self.vehicle)
        print("✅ Scenario loaded")
        
        # Attach sensors
        self.vehicle.sensors.attach('electrics', Electrics())
        
        # Create powertrain sensor
        try:
            self.powertrain_sensor = PowertrainSensor('powertrain', self.beamng, self.vehicle)
            print("✅ Sensors attached")
        except Exception as e:
            print(f"⚠️ PowertrainSensor not available: {e}")
            self.powertrain_sensor = None
        
        # Initial steps to stabilize
        for _ in range(10):
            self.beamng.step(1)
            time.sleep(0.05)
        
        print("✅ Simulation ready!")
        return True
    
    def get_motor_data(self):
        """Get motor torque and speed from BeamNG."""
        motor_torque = 0.0
        motor_speed = 0.0
        vehicle_speed = 0.0
        
        # Poll standard sensors
        try:
            self.vehicle.sensors.poll()
            
            if hasattr(self.vehicle.sensors, '_sensors') and 'electrics' in self.vehicle.sensors._sensors:
                electrics = self.vehicle.sensors._sensors['electrics'].data
                if electrics:
                    vehicle_speed = electrics.get('wheelspeed', 0) or 0
        except:
            pass
        
        # Get powertrain data
        if self.powertrain_sensor:
            try:
                pt_data = self.powertrain_sensor.poll()
                if pt_data:
                    # Look for motor data in powertrain
                    for timestamp, components in pt_data.items():
                        if isinstance(components, dict):
                            for comp_name, comp_data in components.items():
                                if 'motor' in comp_name.lower() and isinstance(comp_data, dict):
                                    motor_torque = comp_data.get('outputTorque1', 0) or 0
                                    motor_speed = comp_data.get('outputAV1', 0) or 0
                                    break
            except:
                pass
        
        # Fallback: estimate from vehicle speed if no motor data
        if motor_torque == 0 and vehicle_speed > 0:
            # Rough estimate: assume constant efficiency
            motor_torque = vehicle_speed * 10  # Very rough
            motor_speed = vehicle_speed * 5
        
        return motor_torque, motor_speed, vehicle_speed
    
    def save_state(self):
        """Save current state to file for dashboard."""
        state = self.battery.get_state_dict()
        
        # Add driving status
        state["vehicle_model"] = self.vehicle_model
        state["simulation_running"] = self.running
        
        with open(DATA_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def run(self, duration=None):
        """Run the live simulation."""
        self.running = True
        print("\n" + "="*60)
        print("🔋 BATTERY DIGITAL TWIN - LIVE SIMULATION")
        print("="*60)
        print(f"🚗 Vehicle: {self.vehicle_model}")
        print(f"⚡ Demo Mode: {'ON (10,000x acceleration)' if self.demo_mode else 'OFF (realistic)'}")
        print(f"📊 Dashboard: Run dashboard.py in another terminal")
        print("="*60)
        print("\n🎮 DRIVE THE CAR IN BEAMNG!")
        print("   Use arrow keys or WASD in the BeamNG window")
        print("   Press Ctrl+C to stop simulation\n")
        
        start = time.time()
        update_count = 0
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start) > duration:
                    break
                
                # Step simulation
                self.beamng.step(3)
                
                # Get motor data
                motor_torque, motor_speed, vehicle_speed = self.get_motor_data()
                
                # Update battery state
                self.battery.update(motor_torque, motor_speed, 0.05, vehicle_speed)
                
                # Save state for dashboard
                update_count += 1
                if update_count % 5 == 0:  # Save every 5th update
                    self.save_state()
                
                # Console output every second
                if update_count % 20 == 0:
                    state = self.battery.get_state_dict()
                    power_str = f"{state['power_kw']:+.1f}kW"
                    if state['is_regen']:
                        power_str += " 🔄REGEN"
                    
                    print(f"⚡ Power: {power_str:20s} | "
                          f"SOC: {state['soc']:.1f}% | "
                          f"SOH: {state['soh']:.2f}% | "
                          f"Day {state['simulated_days']:.0f}")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Simulation stopped by user")
        
        self.running = False
        self.save_state()
        
        # Final summary
        state = self.battery.get_state_dict()
        print("\n" + "="*60)
        print("📊 SIMULATION SUMMARY")
        print("="*60)
        print(f"⏱️ Real time: {state['elapsed_seconds']:.1f} seconds")
        print(f"📅 Simulated: {state['simulated_days']:.1f} days ({state['simulated_years']:.2f} years)")
        print(f"🔋 SOC: 85.0% → {state['soc']:.1f}%")
        print(f"💚 SOH: 100.0% → {state['soh']:.2f}%")
        print(f"🚗 Distance: {state['total_distance_km']:.1f} km")
        print(f"🔄 Cycles: {state['cycle_count']:.1f}")
        print("="*60)
    
    def disconnect(self):
        """Disconnect from BeamNG."""
        if self.beamng:
            try:
                self.beamng.disconnect()
                print("✅ Disconnected from BeamNG")
            except:
                pass


def main():
    print("🔋 EV Battery Digital Twin - Live Demo")
    print("="*50)
    
    # Get user preferences
    print("\n📋 SELECT OPTIONS:")
    print("1. Kia EV3 (sv1ev3) - Default")
    print("2. Tesla Model 3 (Model3_2024)")
    print("3. Custom vehicle")
    
    choice = input("\n🚗 Choose vehicle (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        vehicle = "sv1ev3"
    elif choice == "2":
        vehicle = "Model3_2024"
    elif choice == "3":
        vehicle = input("Enter vehicle model name: ").strip()
    else:
        vehicle = "sv1ev3"
    
    demo = input("\n⚡ Enable demo mode? (Y/n, default=Y): ").strip().lower()
    demo_mode = demo != 'n'
    
    if demo_mode:
        print("\n⏱️ Demo mode: 10,000x time acceleration")
        print("   1 minute of driving = ~7 days simulated")
        print("   15 minutes = ~100 days = ~3 months")
    
    print("\n" + "="*50)
    input("Press Enter to start simulation...")
    
    sim = LiveDemoSimulation(vehicle_model=vehicle, demo_mode=demo_mode)
    
    try:
        if sim.connect():
            sim.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sim.disconnect()


if __name__ == "__main__":
    if not BEAMNG_AVAILABLE:
        print("Please install beamngpy first!")
    else:
        main()
