from fastapi import FastAPI
from pydantic import BaseModel
from lifepo4_model import LFPDynamics
from thermal import LumpedThermal

app = FastAPI()
bat = LFPDynamics()
therm = LumpedThermal()

class StepIn(BaseModel):
    pack_current_A: float
    soc: float
    amb_temp_C: float
    dt_s: float

class StepOut(BaseModel):
    soh: float
    cap_Ah: float
    r_int_ohm: float
    pack_temp_C: float
    max_discharge_kW: float
    max_regen_kW: float

@app.post('/step', response_model=StepOut)
def step(inp: StepIn):
    T_K = inp.amb_temp_C + 273.15
    T_pack = therm.step(inp.pack_current_A, bat.R0, T_K, inp.dt_s)
    bat.step_aging(inp.pack_current_A, inp.soc, T_pack, inp.dt_s)

    base_kW = 200.0 * bat.SOH
    temp_derate = max(0.3, 1.0 - max(0.0, (T_pack - (25+273.15)))/60.0)
    max_discharge_kW = base_kW * temp_derate
    max_regen_kW = 0.5 * max_discharge_kW

    return StepOut(
        soh=bat.SOH,
        cap_Ah=bat.capacity_Ah(),
        r_int_ohm=bat.R0,
        pack_temp_C=T_pack - 273.15,
        max_discharge_kW=max_discharge_kW,
        max_regen_kW=max_regen_kW
    )
