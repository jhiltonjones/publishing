import numpy as np
from magnetic_field_magnitude import magnetic_field, magnetic_moment
from modelling_phi_angle import constant, root_scalar
from jac_controller import theta_angle_solved
import matplotlib.pyplot as plt


def rotate
if __name__ == '__main__':

    mu_0 = 4e-7*np.pi
    B_r = 1.2
    r_epm = 0.03
    len_epm = 0.09
    r = 0.0015
    A_cs = np.pi * r**2
    len_beam = 0.04
    E = 3.6e6
    I = np.pi * r**4/4
    mag = 128e3
    mag_epm = magnetic_moment(B_r, mu_0, r_epm, len_epm)
    p_vec = np.array([0.1,0,0])
    values = np.linspace(np.deg2rad(1), np.deg2rad(80), 10)
    phi_values = np.linspace(np.deg2rad(1), np.deg2rad(80), 4)
    bend_phi = []
    for phi in phi_values:
        bend_values = []
        for k in values:
            mu_hat = np.array([np.cos(k), np.sin(k), 0])
            B = magnetic_field(mu_0, mag_epm, p_vec, mu_hat)
            # print(f"Magnetic field is {B}")
            # print(B[0], phi, mag, A_cs, len_beam, E, I)
            bend_angle = theta_angle_solved(B[0], phi, mag, A_cs, len_beam, E, I)
            print(f"Bending angle is {np.rad2deg(bend_angle)} for phi {np.rad2deg(phi)} and alpha {np.rad2deg(k)}")
            bend_values.append(np.rad2deg(bend_angle))
        bend_phi.append(bend_values)
    plt.figure()
    for phi, bend in zip(phi_values, bend_phi):
        plt.plot(np.rad2deg(values), bend)
        plt.xlabel("Alpha Values")
        plt.ylabel("Bending Angle")
    plt.show()