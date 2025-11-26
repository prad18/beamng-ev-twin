class LumpedThermal:
    def __init__(self, C_th=90000, R_th=0.35, T_init_K=298):
        self.C = C_th     # J/K
        self.R = R_th     # K/W to ambient
        self.T = T_init_K

    def step(self, I, R_int, T_amb_K, dt_s):
        q_joule = (I**2) * R_int
        dT = (q_joule - (self.T - T_amb_K)/self.R) * dt_s / self.C
        self.T += dT
        return self.T
