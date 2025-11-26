import math

class LFPDynamics:
    def __init__(self, q_nom_ah=100.0, r0_milliohm=3.0, soh=1.0):
        self.Q0 = q_nom_ah               # nominal capacity @ BOL
        self.R0 = r0_milliohm/1000       # internal resistance (ohm)
        self.SOH = soh                   # 1.0 = 100%
        self.cycle_loss = 0.0
        self.cal_loss = 0.0

    def arrhenius(self, T_K, Ea=3.5e4):
        R = 8.314
        return math.exp(-Ea/(R*T_K))

    def step_aging(self, I, soc, T_K, dt_s):
        # Cycle aging (simplified): scales with current, DoD, temp
        dod_factor = 1.5*abs(soc-0.5) + 0.25
        k_cycle = 1e-8
        self.cycle_loss += k_cycle * abs(I) * dod_factor * self.arrhenius(T_K) * dt_s

        # Calendar aging (simplified)
        k_cal = 2e-10
        self.cal_loss  += k_cal * self.arrhenius(T_K) * dt_s

        # Combine → SOH
        loss = self.cycle_loss + self.cal_loss
        self.SOH = max(0.6, 1.0 - loss)

        # Resistance growth (toy relation)
        self.R0 = (3e-3) * (1.0 + 2.0*(1.0 - self.SOH))

    def capacity_Ah(self):
        return self.Q0 * self.SOH
