"""
Test script to check what battery data is available from different sensors
"""
import time
import yaml
import json
from datetime import datetime
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, PowertrainSensor

CFG = yaml.safe_load(open(__file__.replace('test_battery_sensors.py', 'config.yaml'), 'r'))

"""
Test script to check what battery data is available from different sensors
"""
import time
import yaml
import json
import os
from datetime import datetime
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Electrics, PowertrainSensor

CFG = yaml.safe_load(open(__file__.replace('test_battery_sensors.py', 'config.yaml'), 'r'))

def test_battery_sensors():
    print("🔋 Battery Sensor Data Analyzer for BeamNG EV Vehicles")
    print("=" * 60)
    print("This tool will analyze battery and powertrain data from any vehicle")
    print("and save organized results for battery degradation testing.\n")
    
    # Get user input for vehicle
    vehicle_model = input("🚗 Enter vehicle model name (e.g., 'sv1ev3', 'cherrier_vivace_ev'): ").strip()
    if not vehicle_model:
        print("❌ No vehicle model specified. Exiting.")
        return
    
    vehicle_name = input(f"🏷️  Enter vehicle name/ID (default: {vehicle_model}): ").strip()
    if not vehicle_name:
        vehicle_name = vehicle_model
    
    print(f"\n🔍 Testing battery sensors for: {vehicle_model}")
    print(f"📁 Results will be saved as: {vehicle_name}")
    
    # Create results directory
    results_dir = "battery_sensor_data"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📂 Created directory: {results_dir}")
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'vehicle_model': vehicle_model,
        'vehicle_name': vehicle_name,
        'test_description': 'Battery sensor data analysis for degradation testing',
        'electrics_data': {},
        'powertrain_data': {},
        'lua_battery_data': {},
        'battery_analysis': {}
    }
    
    # Connect to BeamNG
    beamng = BeamNGpy('localhost', 64256, home=CFG['beamng_home'])
    try:
        beamng.open()
        print("✅ Connected to BeamNG")
    except Exception as e:
        print(f"❌ Failed to connect to BeamNG: {e}")
        print("Make sure BeamNG.tech is running!")
        return
    
    # Create vehicle
    vehicle = Vehicle(vehicle_name, model=vehicle_model)
    print(f"🚗 Created vehicle: {vehicle_model}")
    
    # Create scenario
    scenario_name = f'battery_test_{vehicle_name}_{int(time.time())}'
    scenario = Scenario('smallgrid', scenario_name)
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))
    
    try:
        scenario.make(beamng)
        beamng.load_scenario(scenario)
        beamng.start_scenario()
        beamng.switch_vehicle(vehicle)
        print(f"✅ {vehicle_model} loaded in scenario")
    except Exception as e:
        print(f"❌ Failed to load vehicle {vehicle_model}: {e}")
        beamng.disconnect()
        return
    
    # Attach sensors
    try:
        vehicle.sensors.attach('electrics', Electrics())
        print("✅ Attached electrics sensor")
    except Exception as e:
        print(f"⚠️ Could not attach electrics sensor: {e}")
    
    # PowertrainSensor works differently - it's not attached via the normal sensor system
    powertrain_sensor = None
    try:
        powertrain_sensor = PowertrainSensor('powertrain_test', beamng, vehicle)
        print("✅ Created powertrain sensor (separate from vehicle.sensors)")
    except Exception as e:
        print(f"⚠️ Could not create powertrain sensor: {e}")
    
    # Step simulation and poll sensors
    print("🔄 Initializing sensors...")
    for i in range(5):
        beamng.step(1)
        time.sleep(0.1)
    
    # Poll standard sensors
    try:
        vehicle.sensors.poll()
        print("✅ Polled standard sensors")
    except Exception as e:
        print(f"⚠️ Could not poll standard sensors: {e}")
    
    # Poll powertrain sensor separately if available
    powertrain_data = None
    if powertrain_sensor:
        try:
            powertrain_data = powertrain_sensor.poll()
            print("✅ Polled powertrain sensor successfully")
        except Exception as e:
            print(f"⚠️ Could not poll powertrain sensor: {e}")
    
    print("\n🔍 ELECTRICS SENSOR DATA:")
    print("=" * 40)
    if hasattr(vehicle.sensors, '_sensors') and 'electrics' in vehicle.sensors._sensors:
        electrics_data = vehicle.sensors._sensors['electrics'].data
        if electrics_data:
            # Save all electrics data
            results['electrics_data'] = electrics_data
            
            # Show battery-related fields on screen
            battery_fields = {}
            for key, value in electrics_data.items():
                if any(term in key.lower() for term in ['motor', 'battery', 'energy', 'voltage', 'current', 'power', 'electric']):
                    battery_fields[key] = value
                    print(f"  {key}: {value}")
            
            results['electrics_battery_fields'] = battery_fields
            print(f"\nTotal electrics fields: {len(electrics_data)}")
            print(f"Battery-related fields: {len(battery_fields)}")
        else:
            print("  No electrics data")
            results['electrics_data'] = None
    
    print("\n🔍 POWERTRAIN SENSOR DATA:")
    print("=" * 40)
    if powertrain_data:
        print("Raw powertrain sensor data:")
        results['powertrain_data'] = powertrain_data
        for key, value in powertrain_data.items():
            print(f"  {key}: {value}")
    else:
        print("  No powertrain data available")
        results['powertrain_data'] = None
    
    print("\n🔍 LUA QUERY FOR BATTERY DATA:")
    print("=" * 40)
    try:
        battery_lua = beamng.queue_lua_command("""
            local vehicle = getPlayerVehicle()
            local result = {}
            if vehicle then
                local electrics = vehicle:getElectricsData()
                if electrics then
                    result.motorTorque = electrics.motorTorque
                    result.motorRPM = electrics.motorRPM
                    result.batteryVoltage = electrics.batteryVoltage
                    result.energyLevel = electrics.energyLevel
                    result.wheelspeed = electrics.wheelspeed
                    result.throttle = electrics.throttle
                    result.brake = electrics.brake
                    -- Get all electrics fields
                    result.all_electrics_keys = {}
                    for k, v in pairs(electrics) do
                        table.insert(result.all_electrics_keys, k)
                    end
                end
            end
            return result
        """)
        
        if battery_lua:
            print("Battery-related fields from Lua:")
            results['lua_battery_data'] = battery_lua
            for key, value in battery_lua.items():
                if key != 'all_electrics_keys':
                    print(f"  {key}: {value}")
            
            if 'all_electrics_keys' in battery_lua:
                print(f"\nAll available electrics keys ({len(battery_lua['all_electrics_keys'])}):")
                for key in sorted(battery_lua['all_electrics_keys']):
                    print(f"  {key}")
        else:
            print("  No Lua data returned")
            results['lua_battery_data'] = None
    except Exception as e:
        print(f"  Lua query error: {e}")
    
    beamng.disconnect()
    print("\n✅ Data collection complete")
    
    # Add battery analysis
    results['battery_analysis'] = analyze_battery_data(results)
    
    # Save all results to organized files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results file
    main_file = os.path.join(results_dir, f"{vehicle_name}_battery_analysis_{timestamp}.json")
    with open(main_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save separate detailed files
    if results.get('electrics_data'):
        electrics_file = os.path.join(results_dir, f"{vehicle_name}_electrics_{timestamp}.json")
        with open(electrics_file, 'w') as f:
            json.dump(results['electrics_data'], f, indent=2, default=str)
        print(f"📁 Electrics data: {electrics_file}")
    
    if results.get('powertrain_data'):
        powertrain_file = os.path.join(results_dir, f"{vehicle_name}_powertrain_{timestamp}.json")
        with open(powertrain_file, 'w') as f:
            json.dump(results['powertrain_data'], f, indent=2, default=str)
        print(f"📁 Powertrain data: {powertrain_file}")
    
    if results.get('lua_battery_data'):
        lua_file = os.path.join(results_dir, f"{vehicle_name}_lua_battery_{timestamp}.json")
        with open(lua_file, 'w') as f:
            json.dump(results['lua_battery_data'], f, indent=2, default=str)
        print(f"📁 Lua battery data: {lua_file}")
    
    print(f"📁 Main analysis file: {main_file}")
    print(f"\n🎉 All battery sensor data saved for vehicle: {vehicle_name}")
    return results

def analyze_battery_data(results):
    """Analyze collected data for battery degradation testing suitability."""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'data_sources_available': [],
        'battery_fields_found': [],
        'motor_data_available': False,
        'power_calculation_possible': False,
        'degradation_testing_suitability': 'unknown'
    }
    
    # Check what data sources we have
    if results.get('electrics_data'):
        analysis['data_sources_available'].append('electrics_sensor')
    if results.get('powertrain_data'):
        analysis['data_sources_available'].append('powertrain_sensor')
    if results.get('lua_battery_data'):
        analysis['data_sources_available'].append('lua_queries')
    
    # Check for battery-related fields in electrics
    if results.get('electrics_data'):
        battery_terms = ['motor', 'battery', 'energy', 'voltage', 'current', 'power', 'electric', 'regen']
        for key in results['electrics_data'].keys():
            if any(term in key.lower() for term in battery_terms):
                analysis['battery_fields_found'].append(key)
    
    # Check for motor data in powertrain
    if results.get('powertrain_data'):
        for timestamp_data in results['powertrain_data'].values():
            if isinstance(timestamp_data, dict):
                for component_name, component_data in timestamp_data.items():
                    if 'motor' in component_name.lower() and isinstance(component_data, dict):
                        if 'outputTorque1' in component_data and 'outputAV1' in component_data:
                            analysis['motor_data_available'] = True
                            analysis['power_calculation_possible'] = True
                            break
    
    # Check Lua battery data
    if results.get('lua_battery_data'):
        lua_data = results['lua_battery_data']
        if lua_data.get('motorTorque') is not None or lua_data.get('motorRPM') is not None:
            analysis['motor_data_available'] = True
            analysis['power_calculation_possible'] = True
    
    # Determine degradation testing suitability
    if analysis['power_calculation_possible']:
        analysis['degradation_testing_suitability'] = 'excellent'
    elif analysis['battery_fields_found']:
        analysis['degradation_testing_suitability'] = 'good'
    elif analysis['data_sources_available']:
        analysis['degradation_testing_suitability'] = 'limited'
    else:
        analysis['degradation_testing_suitability'] = 'poor'
    
    return analysis

if __name__ == "__main__":
    print("🔋 BeamNG Battery Sensor Analyzer")
    print("=" * 50)
    
    while True:
        print("\n📋 OPTIONS:")
        print("1. Test a single vehicle")
        print("2. Test multiple vehicles")
        print("3. View saved results")
        print("4. Exit")
        
        choice = input("\n🔢 Choose option (1-4): ").strip()
        
        if choice == "1":
            test_battery_sensors()
        elif choice == "2":
            print("\n🚗 Testing multiple vehicles...")
            vehicles = []
            while True:
                vehicle = input("Enter vehicle model (or 'done' to finish): ").strip()
                if vehicle.lower() == 'done':
                    break
                if vehicle:
                    vehicles.append(vehicle)
            
            for i, vehicle_model in enumerate(vehicles, 1):
                print(f"\n🔄 Testing vehicle {i}/{len(vehicles)}: {vehicle_model}")
                # Temporarily override input for batch processing
                original_input = input
                
                def mock_input(prompt):
                    if "vehicle model name" in prompt.lower():
                        return vehicle_model
                    elif "vehicle name/ID" in prompt.lower():
                        return ""  # Use default
                    return original_input(prompt)
                
                import builtins
                builtins.input = mock_input
                
                try:
                    test_battery_sensors()
                except Exception as e:
                    print(f"❌ Failed to test {vehicle_model}: {e}")
                finally:
                    builtins.input = original_input
                
                if i < len(vehicles):
                    time.sleep(2)  # Brief pause between tests
                    
        elif choice == "3":
            results_dir = "battery_sensor_data"
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
                if files:
                    print(f"\n📁 Found {len(files)} result files in {results_dir}:")
                    for f in sorted(files):
                        print(f"  {f}")
                else:
                    print(f"\n📁 No result files found in {results_dir}")
            else:
                print(f"\n📁 Results directory {results_dir} does not exist")
                
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
