import numpy as np 


def f_xi(xi):
    term1 = (xi + 1)/np.sqrt((xi + 1)**2 +1)
    term2 = (xi -1)/np.sqrt((xi-1)**2+1)
    return 0.5 * (term1 - term2)
def df_dxi(xi):
    a_plus = ((xi +1)**2+1)**1.5
    a_minus= ((xi -1)**2+1)**1.5
    return 0.5 * (1/a_plus - 1/a_minus)
def mag_torque(mag, B):
    return mag * B
def mag_force(mag, B_grad, length):
    return mag * -1*B_grad* length

if __name__ == '__main__':
    Br = 1.45
    R = 0.03
    d = 0.14
    xi = d/R
    print(xi)
    mag = 128e3
    length = 0.01
    B = Br * f_xi(xi)
    dB_dd = (Br/R)* df_dxi(xi)
    print(f"Field is: {B}")
    print(f"Gradient is: {dB_dd}")

    torque = mag_torque(mag, B)
    force = mag_force(mag, dB_dd, length)
    print(f"Magnetic force is: {force}")
    print(f"Magnetic torque is: {torque}")
    print(f"Percentage of force to torque is :{(force/torque)*100}")

    
