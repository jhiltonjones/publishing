import numpy as np
from magnetic_field_magnitude import magnetic_field
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
def constant(B, mag, A_cs, L, E, I):
    return (mag*B*A_cs*L**2)/(E*I)

def integral_cos(phi, theta_L, eps=1e-6):
    def integrand(theta):
        w = np.cos(phi - theta_L) - np.cos(phi - theta)
        if w <= 0:
            return 0.0
        return 1/np.sqrt(w)
    upper = theta_L-eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    xi_val = 0.5*val**2
    return xi_val

def root_theta(rhs_eq, phi, tol = 1e-6):
        eps = 1e-4
        theta_min = eps
        theta_max = max(phi - eps, theta_min)
        def f(theta_l):
            return integral_cos(phi, theta_l) - rhs_eq
        sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
        return sol.root
def tip_angle_from_B_phi_L(B, phi, mag, A_cs, L, E, I):
    print(B, phi, mag, A_cs, L, E, I)
    lam = constant(mag, B, A_cs, L, E, I)
    theta = root_theta(lam, phi)
    return theta  # radians
if __name__ == '__main__':
    length = 0.04
    E = 4.5e6
    radius = 0.0015
    A_cs = np.pi * radius**2
    I = np.pi * radius**4 / 4
    phi = np.deg2rad(60)
    B = 0.02
    mag = 128e3
    rhs_eq = constant(B, mag, A_cs, length, E, I)
    print(rhs_eq)
    theta_L = root_theta(rhs_eq, phi)
    print(np.rad2deg(theta_L))
    angle = tip_angle_from_B_phi_L(B, phi, mag, A_cs, length, E, I)
    print(np.rad2deg(angle))
    lengths = np.linspace(0.04, 0.06, 4)
    phis = np.linspace(np.deg2rad(1), np.pi/2, 30)
    graph_inputs = []

    # for L in lengths:
    #     angles = []
    #     for phi in phis:
    #         rhs_eq = constant(B, mag, A_cs, L, E, I)
    #         theta_L = root_theta(rhs_eq, phi)
    #         angles.append(theta_L)
    #     graph_inputs.append(angles)

    # plt.figure()
    # for L, graph in zip(lengths, graph_inputs):
    #     plt.plot(np.rad2deg(phis), np.rad2deg(graph), label=f"L={L:.2f} m")
    # plt.legend()
    # plt.show()

    
