import numpy as np 
import matplotlib.pyplot as plt

def magnetic_moment(B_r, mu0, r, length):
    term1 = B_r / mu0
    term2 = np.pi * (r**2) * length
    return term1 * term2
def magnetic_field(mu_0, mu, r, mu_hat):
    r_mag = np.linalg.norm(r)
    r_hat = r/np.linalg.norm(r)
    term1 = (mu_0 * mu)/(4* np.pi * r_mag**3)
    term2 = (3*np.outer(r_hat,r_hat))-np.eye(3)
    return ( term1 * term2) @ mu_hat
if __name__ == '__main__':
    mu_0 = 4 * np.pi * 1e-7
    B_r = 1.2
    r = 0.03
    p = 0.09
    m_hat = np.array([-1,0,0])
    x_init = 0.09
    mag_epm = magnetic_moment(B_r, mu_0, r, p)
    r = np.array([0.12,0,0])
    theta = np.pi/2
    mu_hat = np.array([np.cos(theta),np.sin(theta),0])
    magfield = magnetic_field(mu_0, mag_epm, r, mu_hat)
    print(np.linalg.norm(magfield))
    thetas = np.linspace(0, np.pi, 100)
    distances = np.linspace(0.10, 0.12, 5)
    mag_d = []
    for distance in distances:
        r = np.array([distance,0,0])
        field_values = []

        for theta in thetas:
            mu_hat = np.array([np.cos(theta),np.sin(theta),0])
            magfield = magnetic_field(mu_0, mag_epm, r, mu_hat)
            mag_norm = np.linalg.norm(magfield)
            field_values.append(mag_norm)
        mag_d.append(field_values)

    plt.figure()
    for distance ,m in zip(distances, mag_d):
        plt.plot(thetas, m)
    # plt.show()
