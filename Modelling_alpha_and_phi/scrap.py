import numpy as np
from magnetic_field_magnitude import magnetic_field, magnetic_moment
from modelling_phi_angle import constant, root_scalar
from jac_controller import theta_angle_solved
import matplotlib.pyplot as plt

def rotate_about_axis(v, axis, alpha):
    axis = axis / np.linalg.norm(axis)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return (v * c
            + np.cross(axis, v) * s
            + axis * np.dot(axis, v) * (1.0 - c))

if __name__ == '__main__':

    mu_0 = 4e-7 * np.pi
    B_r = 1.2
    r_epm = 0.03
    len_epm = 0.09

    r = 0.0015
    A_cs = np.pi * r**2
    len_beam = 0.04
    E = 4.5e6
    I = np.pi * r**4 / 4
    mag = 128e3

    mag_epm = magnetic_moment(B_r, mu_0, r_epm, len_epm)

    # EPM axis (its central axis and rotation axis): z
    epm_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    # Beam direction: along +x, and beam dipole direction in the undeformed state
    beam_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    m0_hat = beam_dir.copy()  # EPM dipole direction when alpha = 0

    # Distance from EPM centre to beam tip (fixed radius)
    R = 0.135  # m

    # Alpha values (rotation angle of EPM dipole around its own z axis)
    alpha_values = np.linspace(np.deg2rad(1), np.deg2rad(89), 181)

    # Phi values: arc angles of EPM position around the beam tip (in the x–y plane)
    phi_values_deg = [130]
    phi_values = np.deg2rad(phi_values_deg)

    all_Beff = []
    all_Bmag = []
    all_bending = []

    for phi_arc in phi_values:
        # EPM position around the beam tip (beam tip at origin)
        # Beam along +x; phi_arc = 0 ⇒ EPM at (+R, 0, 0)
        p_epm = np.array([R * np.cos(phi_arc), R * np.sin(phi_arc), 0.0])
        # Vector from EPM centre to beam tip
        p_vec = -p_epm

        B_eff_list = []
        B_mag_list = []
        bend_list = []

        for alpha in alpha_values:
            # Rotate EPM dipole around its z-axis
            mu_hat = rotate_about_axis(m0_hat, epm_axis, alpha)

            # Magnetic field at beam tip
            B_vec = magnetic_field(mu_0, mag_epm, p_vec, mu_hat)
            B_mag = np.linalg.norm(B_vec)

            # Effective field component along the beam direction (for plotting/intuition)
            B_eff = np.dot(B_vec, beam_dir)

            # *** Key change: use |B| and the ANGLE between B and the beam ***
            if B_mag > 0:
                cos_phi_eff = B_eff / B_mag
                cos_phi_eff = np.clip(cos_phi_eff, -1.0, 1.0)
                phi_eff = np.arccos(cos_phi_eff)   # angle between B and beam
            else:
                phi_eff = 0.0

            # Use B_mag and phi_eff in the bending model
            bend_angle = theta_angle_solved(B_mag, phi_eff, mag, A_cs, len_beam, E, I)

            B_eff_list.append(B_eff)
            B_mag_list.append(B_mag)
            bend_list.append(np.rad2deg(bend_angle))

        all_Beff.append(np.array(B_eff_list))
        all_Bmag.append(np.array(B_mag_list))
        all_bending.append(np.array(bend_list))

    # ---- 1) Plot B_eff vs alpha for different phi_arc ----
    plt.figure()
    for phi_deg, B_eff_curve in zip(phi_values_deg, all_Beff):
        plt.plot(np.rad2deg(alpha_values), B_eff_curve, label=f"phi_arc = {phi_deg}°")
    plt.xlabel("Alpha (deg)")
    plt.ylabel("B_eff along beam (T)")
    plt.title("Effective magnetic field along beam vs alpha")
    plt.grid(True)
    plt.legend()

    # ---- 2) Plot bending angle vs alpha for different phi_arc ----
    plt.figure()
    for phi_deg, bend_curve in zip(phi_values_deg, all_bending):
        plt.plot(np.rad2deg(alpha_values), bend_curve, label=f"phi_arc = {phi_deg}°")
    plt.xlabel("Alpha (deg)")
    plt.ylabel("Bending angle (deg)")
    plt.title("Beam bending vs alpha for different phi_arc")
    plt.grid(True)
    plt.legend()

    # ---- 3) Plot total |B| vs alpha for different phi_arc ----
    plt.figure()
    for phi_deg, B_mag_curve in zip(phi_values_deg, all_Bmag):
        plt.plot(np.rad2deg(alpha_values), B_mag_curve, label=f"phi_arc = {phi_deg}°")
    plt.xlabel("Alpha (deg)")
    plt.ylabel("|B| (T)")
    plt.title("Total magnetic field magnitude vs alpha")
    plt.grid(True)
    plt.legend()

    plt.show()