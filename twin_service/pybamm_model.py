"""
PyBaMM-based Battery Model for Digital Twin
============================================
Uses Doyle-Fuller-Newman (DFN) electrochemical model for realistic physics.
Integrates with existing FastAPI service and BeamNG simulation.
"""

import pybamm
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PyBaMMBatteryTwin:
    """
    Advanced battery model using PyBaMM electrochemical simulation.
    Provides realistic voltage, temperature, and degradation predictions.
    """
    
    def __init__(
        self,
        chemistry: str = "LFP",  # LiFePO4 for Kia EV3
        capacity_ah: float = 220.0,
        initial_soc: float = 0.8,
        initial_soh: float = 1.0
    ):
        """
        Initialize PyBaMM battery model.
        
        Args:
            chemistry: Battery chemistry (LFP, NMC, NCA)
            capacity_ah: Nominal capacity in Amp-hours
            initial_soc: Starting state of charge (0-1)
            initial_soh: Starting state of health (0-1)
        """
        self.chemistry = chemistry
        self.capacity_ah = capacity_ah
        self.initial_soc = initial_soc
        self.soh = initial_soh
        
        # Degradation tracking
        self.cycle_number = 0
        self.total_throughput_ah = 0
        self.time_elapsed_s = 0
        
        # Temperature tracking
        self.pack_temp_c = 25.0
        
        # Initialize PyBaMM model
        self._setup_model()
        
        # Simulation state
        self.current_solution = None
        self.last_voltage = 3.3 * 100  # ~370V for 100-series pack
        
    def _setup_model(self):
        """Set up PyBaMM model with appropriate chemistry and parameters."""
        
        # Select model based on complexity vs speed tradeoff
        # DFN = most accurate but slower
        # SPM = faster but less accurate
        # SPMe = good balance
        self.model = pybamm.lithium_ion.DFN()
        
        # Select parameter set based on chemistry
        if self.chemistry == "LFP":
            # LiFePO4 parameters (similar to Kia EV3)
            parameter_sets = pybamm.ParameterValues("Ecker2015")
        elif self.chemistry == "NMC":
            parameter_sets = pybamm.ParameterValues("Chen2020")
        elif self.chemistry == "NCA":
            parameter_sets = pybamm.ParameterValues("Marquis2019")
        else:
            # Default to LFP
            parameter_sets = pybamm.ParameterValues("Ecker2015")
        
        # Override capacity to match actual battery
        # PyBaMM uses Ah, we scale the cell capacity
        parameter_sets["Nominal cell capacity [A.h]"] = self.capacity_ah / 100  # Assume 100 cells in series
        
        # Set initial SOC
        parameter_sets["Initial concentration in negative electrode [mol.m-3]"] = \
            parameter_sets["Maximum concentration in negative electrode [mol.m-3]"] * self.initial_soc
        
        # Apply parameters to model
        self.parameter_sets = parameter_sets
        
        # Create solver
        self.solver = pybamm.CasadiSolver(mode="safe", dt_max=60)
        
        print(f"✅ PyBaMM model initialized: {self.chemistry} chemistry, {self.capacity_ah}Ah")
    
    def step(
        self,
        current_a: float,
        soc: float,
        ambient_temp_c: float,
        dt_s: float
    ) -> Dict:
        """
        Simulate one timestep with PyBaMM.
        
        Args:
            current_a: Pack current in Amperes (positive=discharge, negative=charge)
            soc: Current state of charge (0-1)
            ambient_temp_c: Ambient temperature in Celsius
            dt_s: Timestep duration in seconds
        
        Returns:
            Dictionary with battery state (soh, voltage, temp, resistance, etc.)
        """
        
        # Update time tracking
        self.time_elapsed_s += dt_s
        
        # Create experiment for this timestep
        # PyBaMM uses positive current for discharge
        current_sign = current_a  # Already correct sign from simulation
        
        # Limit extreme currents to prevent solver issues
        current_limited = np.clip(current_sign, -400, 400)  # Max 400A charge/discharge
        
        # Create current profile for this timestep
        if abs(current_limited) < 0.1:
            # Rest step
            experiment = pybamm.Experiment([
                f"Rest for {dt_s} seconds"
            ])
        elif current_limited > 0:
            # Discharge
            experiment = pybamm.Experiment([
                f"Discharge at {abs(current_limited)}A for {dt_s} seconds or until 2.5V"
            ])
        else:
            # Charge
            experiment = pybamm.Experiment([
                f"Charge at {abs(current_limited)}A for {dt_s} seconds or until 4.2V"
            ])
        
        try:
            # Simulate with experiment
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_sets,
                experiment=experiment
            )
            
            # Run simulation
            solution = sim.solve(initial_soc=soc)
            self.current_solution = solution
            
            # Extract results
            voltage = solution["Terminal voltage [V]"].entries[-1]
            temperature = solution["Volume-averaged cell temperature [K]"].entries[-1] - 273.15
            
            # Update state
            self.pack_temp_c = temperature
            self.last_voltage = voltage * 100  # Scale to pack voltage (~370V for 100S)
            
        except Exception as e:
            # Fallback to simple model if PyBaMM fails
            print(f"⚠️ PyBaMM solver failed: {e}")
            voltage = self._estimate_voltage(soc)
            temperature = self._estimate_temperature(current_a, ambient_temp_c, dt_s)
            self.pack_temp_c = temperature
        
        # Update degradation
        self._update_degradation(current_a, soc, temperature, dt_s)
        
        # Calculate internal resistance (increases with degradation)
        r_internal = self._calculate_resistance()
        
        # Calculate power limits based on SOH and temperature
        max_discharge_kw = self._calculate_max_discharge_power(soc, temperature)
        max_regen_kw = self._calculate_max_regen_power(soc, temperature)
        
        return {
            'soh': self.soh,
            'cap_Ah': self.capacity_ah * self.soh,
            'r_int_ohm': r_internal,
            'pack_temp_C': temperature,
            'pack_voltage_V': self.last_voltage,
            'max_discharge_kW': max_discharge_kw,
            'max_regen_kW': max_regen_kw,
            'cycle_count': self.cycle_number,
            'total_throughput_Ah': self.total_throughput_ah
        }
    
    def _estimate_voltage(self, soc: float) -> float:
        """Fallback voltage estimation based on SOC."""
        # LFP voltage curve approximation
        if self.chemistry == "LFP":
            # LFP has flat voltage curve around 3.2-3.3V
            if soc > 0.95:
                v_cell = 3.6
            elif soc > 0.8:
                v_cell = 3.4
            elif soc > 0.2:
                v_cell = 3.25
            elif soc > 0.1:
                v_cell = 3.1
            else:
                v_cell = 2.8
        else:
            # NMC/NCA more linear
            v_cell = 2.5 + 1.7 * soc
        
        return v_cell
    
    def _estimate_temperature(self, current_a: float, ambient_c: float, dt_s: float) -> float:
        """Simple thermal model fallback."""
        # Heat generation
        r_internal = self._calculate_resistance()
        heat_w = (current_a ** 2) * r_internal
        
        # Simple thermal dynamics
        thermal_mass = 50000  # J/K
        cooling_rate = 20  # W/K
        
        temp_rise = (heat_w - cooling_rate * (self.pack_temp_c - ambient_c)) * dt_s / thermal_mass
        new_temp = self.pack_temp_c + temp_rise
        
        return np.clip(new_temp, ambient_c - 5, ambient_c + 40)
    
    def _update_degradation(self, current_a: float, soc: float, temp_c: float, dt_s: float):
        """
        Update battery degradation based on stress factors.
        Based on research from literature.
        """
        
        # Track throughput
        self.total_throughput_ah += abs(current_a * dt_s / 3600)
        
        # Cycle counting (rainflow would be better, but simplified here)
        self.cycle_number = self.total_throughput_ah / (2 * self.capacity_ah)
        
        # Degradation factors
        
        # 1. Cycle aging (SEI growth)
        c_rate = abs(current_a) / self.capacity_ah
        cycle_stress = c_rate ** 0.5  # Square root dependency on C-rate
        
        # 2. Calendar aging (Arrhenius)
        activation_energy = 35000  # J/mol (typical for Li-ion)
        gas_constant = 8.314
        temp_k = temp_c + 273.15
        arrhenius_factor = np.exp(-activation_energy / (gas_constant * temp_k))
        
        # 3. SOC stress (high/low SOC accelerate aging)
        if soc > 0.8:
            soc_stress = 1.5
        elif soc < 0.2:
            soc_stress = 1.3
        else:
            soc_stress = 1.0
        
        # Combined degradation rate
        # Typical capacity fade: 0.5-2% per year at moderate conditions
        base_degradation_rate = 1.5e-9  # per second at reference conditions
        
        total_degradation_rate = base_degradation_rate * \
                                 (1 + cycle_stress) * \
                                 arrhenius_factor * \
                                 soc_stress
        
        # Apply degradation
        capacity_loss = total_degradation_rate * dt_s
        self.soh = max(0.6, self.soh - capacity_loss)  # Minimum 60% SOH
    
    def _calculate_resistance(self) -> float:
        """Calculate internal resistance based on SOH."""
        # Resistance grows as battery degrades
        r_base = 0.05  # Ohms for new battery
        r_current = r_base / (self.soh ** 1.5)  # Nonlinear growth
        return r_current
    
    def _calculate_max_discharge_power(self, soc: float, temp_c: float) -> float:
        """Calculate maximum discharge power based on limits."""
        # Base power from SOH
        base_power = 200.0 * self.soh  # kW
        
        # Temperature derating (cold reduces power)
        if temp_c < 0:
            temp_factor = 0.3
        elif temp_c < 15:
            temp_factor = 0.6 + 0.02 * temp_c
        elif temp_c > 45:
            temp_factor = max(0.5, 1.0 - 0.02 * (temp_c - 45))
        else:
            temp_factor = 1.0
        
        # SOC derating (low SOC reduces power)
        if soc < 0.1:
            soc_factor = 0.3
        elif soc < 0.2:
            soc_factor = 0.6
        else:
            soc_factor = 1.0
        
        return base_power * temp_factor * soc_factor
    
    def _calculate_max_regen_power(self, soc: float, temp_c: float) -> float:
        """Calculate maximum regenerative power."""
        # Base regen power (typically 50% of discharge)
        max_discharge = self._calculate_max_discharge_power(soc, temp_c)
        base_regen = 0.5 * max_discharge
        
        # High SOC limits regen (can't charge when full)
        if soc > 0.95:
            soc_factor = 0.1
        elif soc > 0.85:
            soc_factor = 0.5
        else:
            soc_factor = 1.0
        
        return base_regen * soc_factor
    
    def get_state_summary(self) -> Dict:
        """Get human-readable state summary."""
        return {
            'chemistry': self.chemistry,
            'capacity_ah': self.capacity_ah,
            'soh_percent': self.soh * 100,
            'cycles': self.cycle_number,
            'total_throughput_ah': self.total_throughput_ah,
            'temperature_c': self.pack_temp_c,
            'voltage_v': self.last_voltage,
            'time_hours': self.time_elapsed_s / 3600
        }


# Quick test
if __name__ == "__main__":
    print("🔋 Testing PyBaMM Battery Twin")
    
    # Create battery model
    battery = PyBaMMBatteryTwin(
        chemistry="LFP",
        capacity_ah=220.0,
        initial_soc=0.8
    )
    
    print("\n📊 Initial State:")
    print(battery.get_state_summary())
    
    # Simulate discharge
    print("\n⚡ Simulating 60A discharge for 10 seconds...")
    result = battery.step(
        current_a=60.0,
        soc=0.8,
        ambient_temp_c=25.0,
        dt_s=10.0
    )
    
    print("\n📈 Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n✅ PyBaMM integration test complete!")
