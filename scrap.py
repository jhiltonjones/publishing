import numpy as np 
from scipy.integrate import quad
from scipy.optimize import root_scalar

def lambda_rhs(mag, B_mag, a_cs, leng, E, I):
    # mag = \tilde M_r in the paper
    return (mag * B_mag * a_cs * leng**2) / (E * I)

def xi_integral(phi, theta_l, eps = 1e-6):
    def integrand(theta):
        w = np.cos(phi-theta_l) - np.cos(phi-theta)
        return 1.0 / np.sqrt(w)
    upper = theta_l - eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    xi_val = 0.5 * val**2          # this is 1/2 * ξ^2
    return xi_val

def solve_for_lamba(lambda_target, phi, tol=1e-6):
    eps = 1e-4
    theta_min = eps
    theta_max = max(phi - eps, theta_min)
    def f(theta_l):
        return xi_integral(phi, theta_l) - lambda_target
    sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
    return sol.root   # θ_L

# ---- NEW: X(φ,θL) and Y(φ,θL) integrals ----

def X_integral(phi, theta_l, eps=1e-6):
    def integrand(theta):
        w = np.cos(phi-theta_l) - np.cos(phi-theta)
        return np.cos(theta) / np.sqrt(w)
    upper = theta_l - eps   # avoid singularity at θ = θ_L
    val, _ = quad(integrand, 0.0, upper, limit=200)
    return val

def Y_integral(phi, theta_l, eps=1e-6):
    def integrand(theta):
        w = np.cos(phi-theta_l) - np.cos(phi-theta)
        return np.sin(theta) / np.sqrt(w)
    upper = theta_l - eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    return val

if __name__ == '__main__':
    # material + geometry
    mag   = 128e3        # \tilde M_r
    E_mag = 3.6e6        # E
    r     = 0.0015       # radius
    A_cs  = np.pi * r**2 # A
    I_mag = np.pi * r**4 / 4.0  # I
    L     = 0.05         # length
    phi   = np.deg2rad(30)

    B_mag = 0.023       

    lambda_target = lambda_rhs(mag, B_mag, A_cs, L, E_mag, I_mag)
    print("lambda =", lambda_target)

    theta_L = solve_for_lamba(lambda_target, phi)
    print(f"θ_L = {np.rad2deg(theta_L):.2f} deg")

    X_val = X_integral(phi, theta_L)
    Y_val = Y_integral(phi, theta_L)

    pref = np.sqrt(E_mag * I_mag / (2.0 * mag * B_mag * A_cs))

    delta_x = pref * X_val
    delta_y = pref * Y_val

    print(f"δx = {delta_x:.6f} m")
    print(f"δy = {delta_y:.6f} m") 