from energy_minimisation import *
import numpy as np
L_cat = 0.04   # total catheter length [m]
L1    = L_cat

# external dipole magnitude for EPM pose solver (A·m²)
mu_mag = 283

# direction from workspace to magnet (for single_epm_pose_from_B)
e_r   = np.array([1.0, 0.0, 0.0])

# catheter radius (you can change this if needed)
r1 = 0.001  # [m] 

# external permanent magnet position used in bending simulation
# (x, y, z) in meters – here 7 cm horizontally, 4 cm vertically
r_epm = np.array([0.13, 0.0, 0.0])

# NiTi material
E_niti  = 3e6        # Young's modulus [Pa]
rho_niti = 6700       # density [kg/m³]
nu_niti  = 0.33       # Poisson's ratio (for mechanics, may override in arrays)
gvec     = np.array([0.0, 0.0, 9.8])

# Cross-sectional properties
A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0
EI1  = E_niti * I1

# Mass
m1 = rho_niti * A_cs * L_cat

# Beam axes and segment data
axes    = [np.array([-1.0, 0.0, 0.0], float)]  # bending axis
lengths = np.array([L1], dtype=float)
n       = len(lengths)

# Section arrays (1 segment)
r_arr  = np.array([r1],       dtype=float)
E_arr  = np.array([E_niti],   dtype=float)
nu_arr = np.array([0.49],     dtype=float)  # nearly incompressible

# Mass per unit length and segment masses
lam    = rho_niti * A_cs
masses = lam * lengths

# 6n external wrench vector
f_e = np.zeros(6*n, dtype=float)

# Magnetic linear density along inner IPM
Br  = 1.2
mu0 = 4e-7 * np.pi
D_ipm = 3e-3
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_mag = Br * V_ipm / mu0
mu_ipm_hat = np.array([0.0, 0.0, 1.0])
mu_ipm = mu_ipm_mag * mu_ipm_hat

m_dir_local = np.array([0.0, 0.0, -1.0], float)
eta_mag     = (mu_ipm_mag / L_ipm) * m_dir_local
eta0_list   = [eta_mag.copy()]

# initial guess for segment angles
thetas0 = np.zeros(n, float)
D_epm = 80e-3
L_epm = 90e-3
V_epm = np.pi*(0.5*D_epm)**2*L_epm
mu_epm_mag = Br*V_epm/mu0
mu_epm_hat = np.array([0.0, 0,1])
mu_epm = mu_epm_mag * mu_epm_hat
lengths = np.array([L1], dtype=float)
n = len(lengths)

r_arr  = np.array([r1, r1], dtype=float)
E_arr  = np.array([E_niti, E_niti], dtype=float)
nu_arr = np.array([0.49, 0.49], dtype=float)

# # masses per segment (same density/area)
lam = rho_niti * A_cs
# masses = np.array([lam, lam], dtype=float)
masses = lam * lengths      # mass = density × length

# 6n wrench vector
f_e = np.zeros(6*n, dtype=float)

  

m_dir_local = np.array([0.0, 0.0, -1.0], float)
eta_mag = (mu_ipm_mag / L_ipm) * m_dir_local   # or another physically correct linear density
print(eta_mag)
V_beam = A_cs*L_cat
Mag_test2 = 9.3e3                    # [A/m] from paper
m_dir_local = np.array([0.0, 0.0, -1.0], float)
mu_ipm_mag = (1.4 * V_ipm / mu0) *  m_dir_local    # total dipole moment of IPM [A·m²]
print(mu_ipm_hat)
eta_beam = Mag_test2 * A_cs * m_dir_local   # [A·m]
mu_ipm_scalar = Br * V_ipm / mu0          # 0.315 A·m²
eta_ipm = (mu_ipm_scalar / L_ipm) * m_dir_local 
print("magnetisaiton" ,eta_ipm)  # ≈ 7.875 A·m · dir
eta0_list = [eta_ipm]




thetas0 = np.zeros(n, float)

thetas_target = np.array([np.deg2rad(45)])
B_req, info = magnetic_field_for_theta(
    thetas_target, axes, lengths, masses, eta0_list,
    f_e=f_e, gvec=gvec, E=E_niti, r=r1, nu=nu_arr,
    R_base=None, p_base=None, quad_n=12,
    prefer_dir=None, 
    reg=0.0
)
print(B_req, info)