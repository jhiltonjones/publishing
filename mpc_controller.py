import numpy as np
from scipy.linalg import solve_discrete_are
from mpc_controller_functions import trust_radius


class MPC_controller:
    def __init__(self, *, theta_fn, J_fn, dt = 0.0, Np = 10, w_th = 10.0, w_u = 5e-2, theta_band_deg=10.0, eps_theta_deg = 10.0, h_deg_for_radius = 0.5,
                 trust_region_deg = 180, theta_max_deg = 90, u_max_deg_s = 90, j6_min_rad = -5.0, j6_max_rad = 5.0, rate_limit_deg = 15.0):
        self.theta_fn = theta_fn
        self.J_fn = J_fn
        self.dt = float(dt)
        self.Np = int(Np)
        self.Q = np.array([[w_th]], float)
        self.R = np.array([[w_u]], float)
        self.band_deg = float(theta_band_deg)
        self.eps_theta_deg = float(eps_theta_deg)
        self.h_deg_for_radius = float(h_deg_for_radius)
        self.trust_region = np.deg2rad(trust_region_deg)
        self.theta_max = np.deg2rad(theta_max_deg)
        self.u_max = np.deg2rad(u_max_deg_s)
        self.j6_min = float(j6_min_rad)
        self.j6_max = float(j6_max_rad)
        self.rate_limit = np.deg2rad(rate_limit_deg)
        self.A = np.array([[1.0]])
        self.Qf = None
        self.V_T = None
        self.S_np = np.tril(np.ones((self.Np, self.Np)))*self.dt #This is a ZOH which assumes that the input u will be held constant over its interval

    def set_intial_psi(self, psi0_rad):
        self.psi = float(psi0_rad)
        
    def set_dt(self, new_dt):
        self.dt = float(new_dt)
    def _tv_gain_sequence(self, B_list):
        P_next = self.Qf if self.Qf is not None else solve_discrete_are(self.A, B_list[-1], self.Q, self.R)
        K_seq = [None]* self.dt
        for k in range(self.Np-1, -1, -1):
            Bk = B_list[k]
            S = self.R +Bk.T @ P_next @ Bk
            Kk = -np.linalg.solve(S, Bk.T @ P_next @ self.A)
            K_seq[k] = Kk
            Acl = self.A + Bk @ Kk
            P_next = self.Q + Acl.T @ P_next @ Acl + Kk.T @ self.R @ Kk
        if self.Qf is None:
            self.Qf = P_next
        return K_seq
    
    def _build_horizon_linearisation(self, psi_now):
        if not hasattr(self, "U_prev"):
            self.U_prev = np.zeros(self.Np)
        U_nom = np.roll(self.U_prev, -1) #This shifts the previous control inputs left to reuse the starting guess for our linearisation
        U_nom[-1] = U_nom[-2] if self.Np >1 else U_nom[-1]
        psi_nom = psi_now + self.S_np @ U_nom
        B_list = []
        dpsi_vec = np.zeros(self.dt)

        for i in range(self.Np):
            Ji = float(self.J_fn(psi_nom[i]))
            Ji = np.sign(Ji) * max(abs(Ji), 1e-6)
            B_list.append(np.array([[self.dt*Ji]]))
            dpsi_vec[i] = trust_radius(
                self.J_fn, psi_nom[i],
                h_rad=np.deg2rad(self.h_deg_for_radius),
                eps_theta_rad=np.deg2rad(self.eps_theta_deg),
                Jmin=1e-6, Lmin=1e-6, dpsi_cap=self.trust_region
            )
        return psi_nom, U_nom, B_list, dpsi_vec
