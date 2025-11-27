
import numpy as np



# Beam / catheter
L_cat =0.04
E_niti = 2.8e6        # [Pa]
r1     = 0.001     # [m]
A_cs   = np.pi * r1**2
I1     = np.pi * r1**4 / 4.0

# Inner permanent magnet (IPM)
mu0      = 4e-7 * np.pi
D_ipm    = 0.002
L_ipm    = 0.04
V_ipm    = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_mag = .012 * V_ipm / mu0       # total dipole moment of IPM [A·m²]
M = mu_ipm_mag / V_ipm   # this gives A/m

print(f"magnetisation is {M/1000}")
m_dir_local = np.array([0.0, 0.0, -1.0])
eta_mag     = (mu_ipm_mag / L_ipm) * m_dir_local  # moment per unit length [A·m]
eta0_list   = [eta_mag.copy()]

eta_mag = (mu_ipm_mag / L_ipm) * np.array([1.0])   # scalar magnetization magnitude
eta_norm = float(np.linalg.norm(eta_mag))

def Lambda_from_B(B_mag,L_cat):
    print(M, B_mag, L_cat, E_niti, I1)
    return ((M) * B_mag *  A_cs * L_cat**2) / (E_niti * I1)

value = Lambda_from_B(0.01,0.04)
print(value)