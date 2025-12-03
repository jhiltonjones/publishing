import numpy as np 
from scipy.integrate import quad
from scipy.optimize import root_scalar

def lambda_rhs(mag, B_mag, a_cs, leng, E, I):
    return (mag*B_mag*a_cs*leng**2) / (E*I)
def xi_integral(phi, theta_l, eps = 1e-6):
    def integrand(theta):
        w = np.cos(phi-theta_l) - np.cos(phi-theta)
        return 1/np.sqrt(w)
    upper = theta_l - eps
    val, _ = quad(integrand, 0.0, upper, limit=200)
    xi_val = 0.5*val**2
    return xi_val

def solve_for_lamba(lambda_target, phi, tol=1e-6):
    eps = 1e-4
    theta_min = eps
    theta_max = max(phi - eps, theta_min)
    def f(theta_l):
        return xi_integral(phi, theta_l) - lambda_target
    sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
    return sol.root

if __name__ == '__main__':
    # mag   = 128e3       
    # E_mag = 3.6e6     
    # r     = 200e-6       
    # A_cs  = np.pi * r**2
    # I_mag = np.pi * r**4 / 4.0
    # L     = 0.015        
    #Kims parameters

    mag   = 128e3       
    E_mag = 3.6e6     
    r     = 0.0015      
    A_cs  = np.pi * r**2
    I_mag = np.pi * r**4 / 4.0
    L     = 0.04  
    phi = np.deg2rad(66)


    lambda_target = lambda_rhs(mag,0.025, A_cs, L, E_mag, I_mag )
    print(lambda_target)

    theta_angle = solve_for_lamba(lambda_target, phi)
    print(f"Angle is {np.rad2deg(theta_angle)}")
