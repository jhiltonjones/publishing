import numpy as np
from modelling_phi_angle import constant, root_theta
from magnetic_field_magnitude import magnetic_field, magnetic_moment
def theta_angle_solved(B, phi, mag, A_cs, L, E, I):
    rhs_eq = constant(B, mag, A_cs, L, E, I)
    theta_L = root_theta(rhs_eq, phi)
    return theta_L

def dtheta_dB(B, phi, mag, A_cs, L, E, I, dB=1e-4):
    theta_plus = theta_angle_solved(B+dB, phi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B-dB, phi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dB)

def dtheta_dphi(B, phi, mag, A_cs, L, E, I, dphi=1e-4):
    theta_plus = theta_angle_solved(B, phi+dphi, mag, A_cs, L, E, I)
    theta_minus = theta_angle_solved(B, phi-dphi, mag, A_cs, L, E, I)
    return (theta_plus - theta_minus) / (2*dphi)

def dtheta_dL(B, phi, mag, A_cs, L, E, I, dL=1e-4):
    theta_plus = theta_angle_solved(B, phi, mag, A_cs, L+dL, E, I)
    theta_minus = theta_angle_solved(B, phi, mag, A_cs, L-dL, E, I)
    return (theta_plus - theta_minus) / (2*dL)

def jacobian_controller(theta_des,B_init, phi_init, L_init,
                        mag, A_cs, E, I, L_min = 0.04, L_max=0.06, B_min=0.008, B_max = 0.02,
                        phi_min = np.deg2rad(-80), phi_max = np.deg2rad(80), dt=0.05,
                        Kp = 5.0, Ki = 0.0, Kd = 0.5, 
                        max_iter = 300, damping = 1e-3):


    B = B_init
    phi = phi_init
    L = L_init

    e_init = 0
    e_prev = 0.0
    for k in range(max_iter):
        # print(B, phi, mag, A_cs, L, E, I)
        act_theta = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - act_theta
        e_dot = (e-e_prev)/dt
        e_init += e*dt

        theta_do_cmd = Kp*e + Ki*e_init + Kd * e_dot

        J_B = dtheta_dB(B, phi, mag, A_cs, L, E, I)
        J_L = dtheta_dL(B, phi, mag, A_cs, L, E, I)
        J_phi = dtheta_dphi(B, phi, mag, A_cs, L, E, I)

        if (B <= B_min and theta_do_cmd* J_B < 0):
            J_B = 0.0
        if (B>= B_max and theta_do_cmd*J_B > 0):
            J_B = 0.0

        if (L <= L_min and theta_do_cmd* J_L < 0):
            J_L = 0.0

        if (L>= L_max and theta_do_cmd*J_L > 0):
            J_L = 0.0
        
        if (phi <= phi_min and theta_do_cmd* J_phi < 0):
            J_phi = 0.0
        if (phi>= phi_max and theta_do_cmd*J_phi > 0):
            J_phi = 0.0
        
        JJt = J_B**2 + J_L**2 + J_phi**2
        gain = theta_do_cmd/ (JJt+damping**2)
        dB = J_B * gain
        dL = J_L * gain
        dphi = J_phi * gain 

        B += dB*dt
        L += dL*dt
        phi += dphi*dt

        B = min(max(B, B_min), B_max)
        L = min(max(L, L_min), L_max)
        phi = min(max(phi, phi_min), phi_max)

        theta_update = theta_angle_solved(B, phi, mag, A_cs, L, E, I)
        e = theta_des - theta_update
        if abs(e) <= np.deg2rad(2):
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return B, phi, L, np.rad2deg(theta_update)

def dB_dx(x, mu_0, mu, mu_hat, h=1e-4):
    p_plus = np.array([x+h, 0, 0])
    p_minus = np.array([x-h, 0, 0])

    Bp = magnetic_field(mu_0, mu, p_plus, mu_hat)
    Bm = magnetic_field(mu_0, mu, p_minus, mu_hat)
    return (Bp[0] - Bm[0])/ (2*h)
def solve_mag_pose(field_des, x_init, mu_0, mu, mu_hat, dt = 0.05, Kp = 1, Ki = 0.0, Kd = 0.01, damping = 1e-3, max_iter = 500):
    e_init = 0.0
    e_prev = 0.0
    x = x_init

    for k in range(max_iter):
        p_vec = np.array([x, 0, 0])
        # print(f"parameters are x: {x}, mu: {mu}, mu_hat: {mu_hat}")
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        # print(f"Current field is {field_curr[0]}, desired field is {field_des}")
        e = field_des - field_curr[0]
        e_init +=e*dt
        e_dot = (e-e_prev)*dt
        theta_dot_cmd = Kp*e + Ki*e_init + Kd*e_dot
        J_x = dB_dx(x, mu_0, mu, mu_hat)
        JJt = J_x**2
        gain = theta_dot_cmd / (JJt + damping**2)
        dx = J_x *gain
        x += dx * dt
        p_vec = np.array([x, 0, 0])
        field_curr = magnetic_field(mu_0, mu, p_vec, mu_hat)
        e = field_des - field_curr[0]
        if abs(e) <= 1e-5:
            print(f"Converged in {k} steps")
            break
        e_prev = e
    return x, field_curr


if __name__ == '__main__':
    theta_target = np.deg2rad(22)
    mag = 128e3
    r = 0.0015
    E = 4.5e6
    A_cs = np.pi * r**2
    I = np.pi * r**4/4
    B_init = 0.01; phi_init = np.deg2rad(25); L_init = 0.04
    mu_0 = 4e-7*np.pi
    B_r = 1.2
    r = 0.03
    p = 0.09
    m_hat = np.array([1,0,0])
    x_init = 0.09
    mag_epm = magnetic_moment(B_r, mu_0, r, p)
    print(f"Magnetic moment of EPM is {mag_epm}")
    B_sol, phi_sol, L_sol, theta_sol = jacobian_controller(theta_target, B_init, phi_init, L_init, mag, A_cs, E, I, L_max=0.04)
    print(f"B solution: {B_sol*1000}mT, phi_sol: {np.rad2deg(phi_sol)}, L_sol: {L_sol}, with angle of: {theta_sol}")
    mag_pose, field = solve_mag_pose(B_sol, x_init, mu_0, mag_epm, m_hat)
    print(f"Magnetic distance is {mag_pose} which prodices a field of {field}")