def est_pack_current(power_kw, voltage_nom=360.0, eff=0.94):
    """Estimate pack current from electrical power (kW).
    Positive power = discharge, negative = regen.
    """
    power_w = power_kw * 1000.0
    power_w = power_w/eff if power_w >= 0 else power_w*eff
    return power_w / voltage_nom