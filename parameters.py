import numpy as np
# parameters.py (sketch)
L_cat = 0.200
L1 = 0.1
L2 = 0.1

# catheter radii
r1 = 0.00015
r2 = 0.00065

# NiTi material
E_niti = 83e9
rho_niti = 6700
nu_niti = 0.33 
I1 = np.pi * r1**4 / 4.0
EI1 = E_niti * I1
I2 = np.pi * r2**4 / 4.0
EI2 = E_niti * I2

# magnetization of inner tube / IPM
Br  = 1.4
mu0 = 4e-7*np.pi
D_ipm = 3e-3
L_ipm = 3e-3
V_ipm = np.pi*(0.5*D_ipm)**2*L_ipm
mu_ipm_mag = Br*V_ipm/mu0
mu_ipm_hat = np.array([0.0, 0.562, 0.827])
mu_ipm = mu_ipm_mag * mu_ipm_hat
# EPM parameters + nominal pose
D_epm = 60e-3
L_epm = 50e-3
V_epm = np.pi*(0.5*D_epm)**2*L_epm
mu_epm_mag = Br*V_epm/mu0
mu_epm_hat = np.array([0.0, 0.470, 0.883])
mu_epm = mu_epm_mag * mu_epm_hat
p_epm_nom = np.array([0.0, 0.100, 0.260])
gvec = np.array([0.0, -9.81, 0.0])
A_cs = np.pi * r1**2
# Section properties
A1 = np.pi * r1**2
I1 = np.pi * r1**4 / 4.0
EI1 = E_niti * I1

A2 = np.pi * r2**2
I2 = np.pi * r2**4 / 4.0
EI2 = E_niti * I2

m1 = rho_niti * A1 * L_cat
m2 = rho_niti * A2 * L_cat
m_cat_total = m1 + m2
axes = [np.array([-1.0, 0.0, 0.0], float)]
        # np.array([-1.0, 0.0, 0.0], float)]
