"""
Kia EV3 Battery Twin Simulation
This script uses the Kia EV3 model for realistic EV battery testing
"""
import time, requests, yaml, math, random, json
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Damage, Timer
from current_model import est_pack_current

CFG = yaml.safe_load(open(__file__.replace('kia_ev3_simulation.py', 'config.yaml'), 'r'))
TWIN_URL = CFG.get('twin_url', 'http://127.0.0.1:8008/step')

# Load Kia EV3 configuration if available
KIA_CONFIG_FILE = 'kia_ev3_config.json'

def load_kia_config():
    """Load Kia EV3 configuration from previous exploration"""
    try:
        with open(KIA_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ Kia EV3 config not found. Run explore_kia_ev3.py first.")
        return None

def twin_step(I, soc, ambC, dt):
    """Call the battery twin service"""
    try:
        r = requests.post(TWIN_URL, json={
            "pack_current_A": float(I),
            "soc": float(soc),
            "amb_temp_C": float(ambC),
            "dt_s": float(dt)
        }, timeout=2.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Twin service error: {e}")
        return {
            'soh': 1.0,
            'cap_Ah': 100.0,  # Kia EV3 has ~81.4 kWh battery
            'r_int_ohm': 0.1,
            'pack_temp_C': 25.0,
            'max_discharge_kW': 200.0,  # Kia EV3 can do ~150kW charging
            'max_regen_kW': 100.0
        }

def get_kia_model_name():
    """Get the correct Kia EV3 model name"""
    # Use the correct name you provided
    possible_names = [
        'sv1ev3',           # Your confirmed working name
    ]
    
    for name in possible_names:
        try:
            vehicle = Vehicle('test', model=name)
            print(f"✅ Found working Kia EV3 model: {name}")
            return name
        except:
            continue
    
    print("❌ Could not find Kia EV3 model")
    return None

def setup_kia_ev3_vehicle():
    """Setup the Kia EV3 vehicle with optimal configuration"""
    model_name = get_kia_model_name()
    
    if not model_name:
        raise Exception("Kia EV3 model not found")
    
    vehicle = Vehicle('kia_ev3', model=model_name)
    
    # Try to configure for optimal electric performance
    try:
        vehicle.queue_lua_command("""
            local vehicle = be:getPlayerVehicle(0)
            if vehicle then
                -- Set to maximum performance mode if available
                local powertrain = vehicle:getPowertrain()
                if powertrain then
                    -- Enable sport mode or max performance
                    powertrain:setMode("sport")
                end
            end
        """)
        print("🏁 Configured Kia EV3 for performance mode")
    except:
        print("ℹ️ Using default Kia EV3 configuration")
    
    return vehicle, model_name

def run_kia_ev3_simulation(test_duration=2000, driving_pattern='city'):
    """Run battery simulation with Kia EV3"""
    print("🚗 Starting Kia EV3 Battery Twin Simulation...")
    
    # Load configuration if available
    kia_config = load_kia_config()
    if kia_config:
        print(f"📋 Using saved Kia EV3 configuration")
        model_name = kia_config['model_name']
    else:
        print("📋 No saved config, discovering Kia EV3...")
    
    # Connect to BeamNG
    beamng = BeamNGpy('localhost', 64256, home=CFG['beamng_home'])
    beamng.open()
    print("✅ Connected to BeamNG")
    
    # Setup Kia EV3
    try:
        vehicle, model_name = setup_kia_ev3_vehicle()
        print(f"✅ Created Kia EV3: {model_name}")
    except Exception as e:
        print(f"❌ Failed to setup Kia EV3: {e}")
        return
    
    # Create scenario
    scenario_name = f'kia_ev3_test_{random.randint(1000, 9999)}'
    scenario = Scenario('smallgrid', scenario_name)
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))
    scenario.make(beamng)
    print(f"✅ Created scenario: {scenario_name}")
    
    # Load and start
    beamng.load_scenario(scenario)
    beamng.start_scenario()
    beamng.switch_vehicle(vehicle)
    print("✅ Scenario loaded and started")
    
    # Add sensors
    vehicle.sensors.attach('electrics', Electrics())
    vehicle.sensors.attach('damage', Damage())
    vehicle.sensors.attach('timer', Timer())
    print("✅ Sensors attached")
    
    # Set up AI driving for realistic behavior
    vehicle.ai_set_mode('span')
    vehicle.ai_set_speed(20)  # 20 m/s = ~45 mph
    vehicle.ai_set_waypoint('next')
    print("🤖 AI driver activated")
    
    # Initialize battery state (Kia EV3 specifications)
    soc = 0.8  # Start at 80%
    pack_capacity_Ah = 220.0  # Kia EV3 ~81.4kWh / 370V = ~220Ah
    ambient_temp_C = 25.0
    t_prev = time.time()
    
    # Data logging
    simulation_data = []
    
    print(f"\n🔋 Starting Kia EV3 Battery Simulation ({driving_pattern} pattern)")
    print("Step | SOC   | Current | Power  | SOH    | Pack°C | Motor°C | Speed | Efficiency")
    print("-" * 85)
    
    for step in range(test_duration):
        # Get real vehicle telemetry
        vehicle.sensors.poll()
        
        # Try different methods to get electrics data
        electrics = {}
        try:
            # Method 1: Try _sensors attribute
            if hasattr(vehicle.sensors, '_sensors') and 'electrics' in vehicle.sensors._sensors:
                electrics = vehicle.sensors._sensors['electrics'].data
            # Method 2: Use Lua query for vehicle electrics
            else:
                electrics_lua = vehicle.query_lua("""
                    local vehicle = be:getPlayerVehicle(0)
                    local result = {}
                    if vehicle then
                        -- Get electrics data directly from vehicle
                        result.motorTorque = vehicle.electrics.motorTorque or 0
                        result.motorRPM = vehicle.electrics.motorRPM or 0
                        result.wheelspeed = vehicle.electrics.wheelspeed or 0
                        result.batteryVoltage = vehicle.electrics.batteryVoltage or 370
                        result.energyLevel = vehicle.electrics.energyLevel or 0.8
                        result.throttle = vehicle.electrics.throttle or 0
                        result.brake = vehicle.electrics.brake or 0
                        result.rpm = vehicle.electrics.rpm or 0
                        result.speed = vehicle.electrics.airspeed or 0
                        
                        -- Try to get powertrain info
                        local powertrain = vehicle:getPowertrain()
                        if powertrain then
                            result.powertrainTorque = powertrain:getTorque() or 0
                            result.powertrainRPM = powertrain:getRPM() or 0
                        end
                    end
                    return result
                """)
                if electrics_lua:
                    electrics = electrics_lua
        except Exception as e:
            print(f"Sensor access method failed: {e}")
            electrics = {}
        
        # Real Kia EV3 telemetry
        wheel_speed = electrics.get('wheelspeed', 0)  # m/s
        motor_torque = electrics.get('motorTorque', 0)  # Nm
        motor_rpm = electrics.get('motorRPM', 0)
        battery_voltage = electrics.get('batteryVoltage', 370.0)  # Kia EV3 ~370V
        throttle_input = electrics.get('throttle', 0)
        brake_input = electrics.get('brake', 0)
        energy_level = electrics.get('energyLevel', soc)  # If available
        
        # Calculate power from real telemetry
        if motor_rpm > 10:
            motor_rps = motor_rpm / 60.0
            mech_power_kw = abs(motor_torque * motor_rps) / 1000.0
            
            # Kia EV3 motor efficiency curve (approximate)
            efficiency = 0.85 + 0.1 * (1 - abs(motor_torque) / 300.0)  # 85-95% efficiency
            
            if motor_torque >= 0:  # Motoring
                elec_power_kw = mech_power_kw / efficiency
            else:  # Regenerating
                elec_power_kw = -mech_power_kw * 0.8  # 80% regen efficiency
        else:
            mech_power_kw = 0.0
            elec_power_kw = 0.0
            efficiency = 0.9
        
        # Apply driving patterns for realistic testing
        if driving_pattern == 'city':
            # City driving: stop and go
            cycle = step % 200
            if cycle < 40:  # Acceleration
                target_power = 80.0
                vehicle.control(throttle=0.7, brake=0.0)
            elif cycle < 120:  # Cruise
                target_power = 25.0
                vehicle.control(throttle=0.3, brake=0.0)
            elif cycle < 160:  # Regen braking
                target_power = -60.0
                vehicle.control(throttle=0.0, brake=0.6)
            else:  # Stop
                target_power = 5.0
                vehicle.control(throttle=0.0, brake=0.0)
                
        elif driving_pattern == 'highway':
            # Highway driving: sustained speed
            cycle = step % 600
            if cycle < 100:  # Acceleration to highway speed
                target_power = 120.0
                vehicle.control(throttle=0.8, brake=0.0)
            elif cycle < 500:  # Highway cruise
                target_power = 35.0
                vehicle.control(throttle=0.4, brake=0.0)
            else:  # Deceleration
                target_power = -30.0
                vehicle.control(throttle=0.0, brake=0.4)
                
        elif driving_pattern == 'aggressive':
            # Aggressive driving: high performance
            cycle = step % 300
            if cycle < 80:  # Hard acceleration
                target_power = 150.0  # Near Kia EV3 max power
                vehicle.control(throttle=1.0, brake=0.0)
            elif cycle < 180:  # High speed
                target_power = 60.0
                vehicle.control(throttle=0.6, brake=0.0)
            elif cycle < 220:  # Hard braking
                target_power = -100.0
                vehicle.control(throttle=0.0, brake=0.8)
            else:  # Recovery
                target_power = 10.0
                vehicle.control(throttle=0.1, brake=0.0)
        
        # Use actual power if significant, otherwise use pattern target
        if abs(elec_power_kw) > 5.0:
            final_power_kw = elec_power_kw
        else:
            final_power_kw = target_power
        
        # Environmental variations
        ambient_temp_C = 25.0 + 15.0 * math.sin(step * 0.003)  # Slow temperature cycle
        
        # Add realistic events
        if step % 1000 < 50:  # Fast charging simulation
            final_power_kw = -150.0  # 150kW DC fast charging
            print(f"⚡ DC Fast Charging: {final_power_kw}kW")
        elif step % 1500 < 20:  # Highway hill climb
            final_power_kw = max(final_power_kw, 100.0)
            ambient_temp_C = 35.0  # Hot conditions
        
        # Calculate battery current
        current_A = est_pack_current(final_power_kw)
        
        # Time step
        now = time.time()
        dt = max(0.02, now - t_prev)
        t_prev = now
        
        # Update SOC
        coulombs = current_A * dt
        ah_change = coulombs / 3600.0
        soc = max(0.05, min(0.95, soc - ah_change / pack_capacity_Ah))
        
        # Call battery twin
        twin_result = twin_step(current_A, soc, ambient_temp_C, dt)
        
        # Update capacity
        pack_capacity_Ah = twin_result.get('cap_Ah', pack_capacity_Ah)
        
        # Estimate motor temperature (simple model)
        motor_temp_C = ambient_temp_C + abs(motor_torque) * 0.05
        
        # Log data
        data_point = {
            'step': step,
            'time': time.time(),
            'soc': soc,
            'current_A': current_A,
            'power_kw': final_power_kw,
            'wheel_speed_ms': wheel_speed,
            'motor_torque_nm': motor_torque,
            'motor_rpm': motor_rpm,
            'efficiency': efficiency,
            'pack_temp_C': twin_result['pack_temp_C'],
            'motor_temp_C': motor_temp_C,
            'soh': twin_result['soh'],
            'ambient_temp_C': ambient_temp_C
        }
        simulation_data.append(data_point)
        
        # Print results every 50 steps
        if step % 50 == 0:
            print(f"{step:4d} | {soc:.3f} | {current_A:7.1f} | {final_power_kw:6.1f} | "
                  f"{twin_result['soh']:.4f} | {twin_result['pack_temp_C']:6.1f} | "
                  f"{motor_temp_C:7.1f} | {wheel_speed*3.6:5.1f} | {efficiency:.3f}")
        
        # Safety checks
        if soc < 0.1:
            print("🔋 Battery critically low - ending simulation")
            break
        if twin_result['pack_temp_C'] > 55.0:
            print("🌡️ Battery overheated - ending simulation")
            break
        
        # Small delay
        time.sleep(0.05)
    
    beamng.close()
    
    # Save results
    results_file = f'kia_ev3_results_{driving_pattern}_{int(time.time())}.json'
    with open(results_file, 'w') as f:
        json.dump(simulation_data, f, indent=2)
    
    # Analysis
    analyze_kia_results(simulation_data, driving_pattern)
    
    print(f"\n📁 Results saved to: {results_file}")
    print("✅ Kia EV3 simulation completed!")

def analyze_kia_results(data, pattern):
    """Analyze the Kia EV3 simulation results"""
    if not data:
        return
    
    initial_soh = data[0]['soh']
    final_soh = data[-1]['soh']
    soh_degradation = initial_soh - final_soh
    
    max_power = max(d['power_kw'] for d in data)
    min_power = min(d['power_kw'] for d in data)
    avg_efficiency = sum(d['efficiency'] for d in data) / len(data)
    max_speed = max(d['wheel_speed_ms'] for d in data) * 3.6  # Convert to km/h
    
    energy_consumed = sum(d['power_kw'] * 0.05 / 3600 for d in data if d['power_kw'] > 0)  # kWh
    energy_regen = abs(sum(d['power_kw'] * 0.05 / 3600 for d in data if d['power_kw'] < 0))  # kWh
    
    print("\n" + "=" * 60)
    print("📊 KIA EV3 SIMULATION ANALYSIS")
    print("=" * 60)
    print(f"Driving Pattern: {pattern.upper()}")
    print(f"SOH Degradation: {soh_degradation:.6f} ({soh_degradation*100:.4f}%)")
    print(f"Max Discharge Power: {max_power:.1f} kW")
    print(f"Max Regen Power: {abs(min_power):.1f} kW")
    print(f"Average Motor Efficiency: {avg_efficiency:.1%}")
    print(f"Maximum Speed: {max_speed:.1f} km/h")
    print(f"Energy Consumed: {energy_consumed:.2f} kWh")
    print(f"Energy Regenerated: {energy_regen:.2f} kWh")
    print(f"Regen Efficiency: {(energy_regen/energy_consumed)*100:.1f}%")
    print("=" * 60)

def main():
    print("🚗 Kia EV3 Battery Twin Simulation")
    print("=" * 50)
    
    print("Choose driving pattern:")
    print("1. City driving (stop-and-go)")
    print("2. Highway driving (sustained speed)")
    print("3. Aggressive driving (performance test)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    pattern_map = {'1': 'city', '2': 'highway', '3': 'aggressive'}
    pattern = pattern_map.get(choice, 'city')
    
    duration = input("Enter test duration in steps (default 2000): ").strip()
    try:
        duration = int(duration) if duration else 2000
    except:
        duration = 2000
    
    print(f"\n🎯 Running {pattern} pattern for {duration} steps...")
    run_kia_ev3_simulation(test_duration=duration, driving_pattern=pattern)

if __name__ == "__main__":
    main()