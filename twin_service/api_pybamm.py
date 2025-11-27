"""
FastAPI Battery Twin Service with PyBaMM Integration
====================================================
Enhanced battery physics service using electrochemical modeling.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import PyBaMM model
try:
    from pybamm_model import PyBaMMBatteryTwin
    PYBAMM_AVAILABLE = True
except ImportError:
    print("⚠️ PyBaMM not available, falling back to simple model")
    from lifepo4_model import LFPDynamics
    from thermal import LumpedThermal
    PYBAMM_AVAILABLE = False

app = FastAPI(
    title="EV Battery Digital Twin with PyBaMM",
    description="Advanced battery simulation using electrochemical models",
    version="2.0.0"
)

# Initialize battery model
if PYBAMM_AVAILABLE:
    battery_model = PyBaMMBatteryTwin(
        chemistry="LFP",
        capacity_ah=220.0,
        initial_soc=0.8,
        initial_soh=1.0
    )
    print("✅ Using PyBaMM electrochemical model")
else:
    # Fallback to simple model
    battery_model = LFPDynamics(q_nom_ah=220.0)
    thermal_model = LumpedThermal()
    print("⚠️ Using simplified physics model")


class BatteryRequest(BaseModel):
    """Request schema for battery timestep."""
    pack_current_A: float
    soc: float
    amb_temp_C: float
    dt_s: float
    accelerated_dt_s: Optional[float] = None  # For accelerated degradation


class BatteryResponse(BaseModel):
    """Response schema with battery state."""
    soh: float
    cap_Ah: float
    r_int_ohm: float
    pack_temp_C: float
    max_discharge_kW: float
    max_regen_kW: float
    
    # Additional PyBaMM fields
    pack_voltage_V: Optional[float] = None
    cycle_count: Optional[float] = None
    total_throughput_Ah: Optional[float] = None
    model_type: str = "simple"


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "EV Battery Digital Twin",
        "version": "2.0.0",
        "model": "PyBaMM" if PYBAMM_AVAILABLE else "Simple",
        "status": "ready"
    }


@app.get("/info")
async def info():
    """Get battery model information."""
    if PYBAMM_AVAILABLE:
        return {
            "model_type": "PyBaMM Electrochemical",
            "chemistry": battery_model.chemistry,
            "capacity_ah": battery_model.capacity_ah,
            "state": battery_model.get_state_summary()
        }
    else:
        return {
            "model_type": "Simplified Physics",
            "chemistry": "LFP",
            "capacity_ah": 220.0
        }


@app.post("/step", response_model=BatteryResponse)
async def battery_step(request: BatteryRequest):
    """
    Process one battery simulation timestep.
    
    Args:
        request: Battery input parameters (current, SOC, temperature, timestep)
    
    Returns:
        Battery state after timestep (SOH, capacity, temperature, limits)
    """
    
    try:
        # Use accelerated dt for degradation if provided, otherwise use real dt
        degradation_dt = request.accelerated_dt_s if request.accelerated_dt_s else request.dt_s
        
        if PYBAMM_AVAILABLE:
            # Use PyBaMM model with separate real and accelerated timesteps
            result = battery_model.step(
                current_a=request.pack_current_A,
                soc=request.soc,
                ambient_temp_c=request.amb_temp_C,
                dt_s=request.dt_s,  # Real timestep for electrochemistry
                degradation_dt_s=degradation_dt  # Accelerated timestep for aging
            )
            
            return BatteryResponse(
                soh=result['soh'],
                cap_Ah=result['cap_Ah'],
                r_int_ohm=result['r_int_ohm'],
                pack_temp_C=result['pack_temp_C'],
                max_discharge_kW=result['max_discharge_kW'],
                max_regen_kW=result['max_regen_kW'],
                pack_voltage_V=result.get('pack_voltage_V'),
                cycle_count=result.get('cycle_count'),
                total_throughput_Ah=result.get('total_throughput_Ah'),
                model_type="PyBaMM"
            )
            
        else:
            # Use simple model
            T_K = request.amb_temp_C + 273.15
            T_pack = thermal_model.step(
                request.pack_current_A,
                battery_model.R0,
                T_K,
                request.dt_s
            )
            
            battery_model.step_aging(
                request.pack_current_A,
                request.soc,
                T_pack,
                request.dt_s
            )
            
            base_kW = 200.0 * battery_model.SOH
            temp_derate = max(0.3, 1.0 - max(0.0, (T_pack - (25+273.15)))/60.0)
            max_discharge_kW = base_kW * temp_derate
            max_regen_kW = 0.5 * max_discharge_kW
            
            return BatteryResponse(
                soh=battery_model.SOH,
                cap_Ah=battery_model.capacity_Ah(),
                r_int_ohm=battery_model.R0,
                pack_temp_C=T_pack - 273.15,
                max_discharge_kW=max_discharge_kW,
                max_regen_kW=max_regen_kW,
                model_type="Simple"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Battery simulation error: {str(e)}")


@app.post("/reset")
async def reset_battery(soc: float = 0.8, soh: float = 1.0):
    """Reset battery to initial conditions."""
    global battery_model, thermal_model
    
    if PYBAMM_AVAILABLE:
        battery_model = PyBaMMBatteryTwin(
            chemistry="LFP",
            capacity_ah=220.0,
            initial_soc=soc,
            initial_soh=soh
        )
    else:
        battery_model = LFPDynamics(q_nom_ah=220.0, soh=soh)
        thermal_model = LumpedThermal()
    
    return {"status": "reset", "soc": soc, "soh": soh}


if __name__ == "__main__":
    print("🚀 Starting Battery Twin Service with PyBaMM")
    print("="*60)
    
    if PYBAMM_AVAILABLE:
        print("✅ PyBaMM electrochemical model loaded")
        print(f"   Chemistry: {battery_model.chemistry}")
        print(f"   Capacity: {battery_model.capacity_ah}Ah")
    else:
        print("⚠️ Using simplified physics model")
        print("   Install PyBaMM: pip install pybamm")
    
    print("\n📡 API Endpoints:")
    print("   GET  /          - API info")
    print("   GET  /info      - Battery model details")
    print("   POST /step      - Simulate timestep")
    print("   POST /reset     - Reset battery state")
    print("\n🌐 Starting server on http://127.0.0.1:8008")
    print("="*60)
    
    uvicorn.run(app, host="127.0.0.1", port=8008)
