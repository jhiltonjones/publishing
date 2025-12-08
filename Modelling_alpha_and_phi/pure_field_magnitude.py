import numpy as np 
import matplotlib.pyplot as plt
def pure_magnitude(alpha, theta_0):
    term = np.sqrt(1+3*(np.cos(theta_0 - alpha)**2))
    return term

if __name__ == '__main__':
    theta_0= np.pi/3
    alpha = np.pi
    mu = np.array([np.cos(alpha), np.sin(alpha), 0])
    field = pure_magnitude(alpha, theta_0)
    print(f"field: {field}")
    alpha_values = np.linspace(0,np.pi, 100)
    phase_shifts = [np.pi/6, np.pi/3, np.pi/2]
    # field_values = []

    phase_shifts_val = []
    for phase in phase_shifts:
        field_values = []
        for alpha in alpha_values:
            mu = np.array([np.cos(alpha), np.sin(alpha), 0])
            field = pure_magnitude(alpha, phase)
            field_values.append(field)
        phase_shifts_val.append(field_values)
    plt.figure()
    for phase, i in zip(phase_shifts, phase_shifts_val):
        plt.plot(alpha_values, i)
    plt.xlabel("Alpha Rotation")
    plt.ylabel("Field Magnitude")
    plt.title("Alpha Rotation vs Field Magnitude")
    plt.show()
        