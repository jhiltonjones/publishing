import numpy as np

def magnetic_field_bar(mu0, mag, p_vec, m_hat):
    r = np.linalg.norm(p_vec)
    p_hat = p_vec/r
    constant = (mu0*mag)/(4*np.pi*r**3)
    term1 = (3* np.outer(p_hat, p_hat)) - np.eye(3)
    return (constant*term1)@m_hat
def z_axis_rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta),0],
              [0,0,1]])
    return 
def magnetic_moment(B_r, r, length):
    term1 = B_r/mu0
    term2 = np.pi*(r**2)*length
    return term1*term2

I3 = np.eye(3)
p_vec= np.array([0.084,0,0])
m_vec = np.array([-1,0,0])
mu0 = 4e-7*np.pi
mag_hat = np.array([-1,0,0])
theta = np.pi/2
R = z_axis_rot(theta)
m_hat = R@mag_hat

values = np.linspace(0,2*np.pi/2,5)
mag_mom = magnetic_moment(1.2, 0.03, 0.09)
print(mag_mom)
mf = magnetic_field_bar(mu0, mag_mom,p_vec, m_hat)
print(mf[1]*1000)
# for value in values:
#     R = z_axis_rot(value)
#     m_hat = R @mag_hat
#     mf = magnetic_field_bar(mu0, mag_mom,p_vec, m_hat)
#     print(f"Magnetic field is {np.linalg.norm(mf)} at theta value of {np.rad2deg(value)}")