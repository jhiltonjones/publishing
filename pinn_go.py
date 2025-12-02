# import tensorflow as tf
# import deepxde as dde
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

# Beam / catheter
L_cat =0.024
E_niti = 2.8e6         # [Pa]
r1     = 0.0005        # [m]
A_cs   = np.pi * r1**2
I1     = np.pi * r1**4 / 4.0

# Inner permanent magnet (IPM)
Br       = 1.4
mu0      = 4e-7 * np.pi
D_ipm    = 0.002
L_ipm    = 0.04
V_ipm    = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_mag = 0.012 * V_ipm / mu0       # total dipole moment of IPM [A·m²]

m_dir_local = np.array([0.0, 0.0, -1.0])
eta_mag     = (mu_ipm_mag / L_ipm) * m_dir_local  # moment per unit length [A·m]
eta0_list   = [eta_mag.copy()]
mag_test = mu_ipm_mag / V_ipm
print(f"mag_test is {mag_test}")
def ddy(x, y):
    return dde.grad.hessian(y, x)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# --------- Dimensionless Λ as in the paper ----------
eta_mag = (mu_ipm_mag / L_ipm) * np.array([1.0])   # scalar magnetization magnitude
eta_norm = float(np.linalg.norm(eta_mag))

def Lambda_from_B(B_mag):
    print(mag_test, B_mag, L_cat, E_niti, I1)
    return (mag_test * B_mag * A_cs * L_cat**2) / (E_niti * I1)



# --------- PINN for θ(ŝ), not u(x) ----------

def make_theta_pde(Lambda, phi_B):
    """Returns a PDE function enforcing θ'' + Λ sin(φ - θ) = 0."""
    Lambda_tf = tf.constant(Lambda, dtype=tf.float32)
    phi_tf    = tf.constant(phi_B,   dtype=tf.float32)

    def pde(x, y):
        theta    = y
        theta_xx = dde.grad.hessian(theta, x)     # d²θ/dŝ²

        # NOTE: +, not -
        return theta_xx + Lambda_tf * tf.sin(phi_tf - theta)

    return pde



def solve_theta_for_field(B_mag, phi_B, geom, iters=8000):
    """
    Train a PINN for θ(ŝ) on ŝ ∈ [0,1], given a physical field B and direction φ.
    """

    Lambda = Lambda_from_B(B_mag)
    print(f"Λ (dimensionless) = {Lambda}")

    pde = make_theta_pde(Lambda, phi_B)

    # Boundary conditions for θ:
    # Clamped base: θ(0) = 0
    # Free tip:     θ'(1) = 0  (no bending moment)
    def boundary_l(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0)

    def boundary_r(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1)

    bc_theta0 = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_l)  # θ(0)=0
    bc_thetaL = dde.icbc.NeumannBC( geom, lambda x: 0.0, boundary_r)   # θ'(1)=0

    data = dde.data.PDE(
        geom,
        pde,
        [bc_theta0, bc_thetaL],
        num_domain=80,
        num_boundary=20,
        num_test=100,
    )

    layer_size = [1] + [50] * 4 + [1]
    net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=0.001,
        loss_weights=[1.0, 50.0, 50.0]   # PDE, θ(0), θ'(1)
    )
    model.train(iterations=20000)
    model.compile("L-BFGS")
    model.train()


    # Evaluate θ on a grid of ŝ
    s_hat = np.linspace(0, 1, 400)[:, None]
    theta_pred = model.predict(s_hat).flatten()

    return s_hat.flatten(), theta_pred




geom = dde.geometry.Interval(0, 1)
angle  = np.linspace(0, np.pi/2, 6)
# skip 0 and π/2 for analytic
field_cases = [(0.0, 0.0)] + [(0.04, phi) for phi in angle[1:-1]]


# plt.figure(figsize=(7, 5))
# print((eta_norm * 0.01*L_cat**2 / (E_niti * I1)))


def xi_integral(phi, theta_L, eps=1e-6):
    """
    ξ(φ, θ_L) = ∫_0^{θ_L} [cos(φ - θ_L) - cos(φ - θ)]^(-1/2) dθ
    Eq. (2.76) in the paper.
    """
    def integrand(theta):
        w = np.cos(phi - theta_L) - np.cos(phi - theta)
        return 1.0 / np.sqrt(w)

    # avoid the square-root singularity exactly at θ = θ_L
    upper = float(theta_L - eps)
    val, _ = quad(integrand, 0.0, upper, limit=200)
    return val


def Lambda_from_thetaL(phi, theta_L):
    """
    Λ(φ, θ_L) = (M̃_r B A L^2 / EI) = 0.5 * ξ(φ, θ_L)^2
    Eq. (2.77).
    """
    xi_val = xi_integral(phi, theta_L)
    return 0.5 * xi_val**2


def solve_theta_L_for_Lambda(Lambda_target, phi, tol=1e-6):
    """
    Given Λ and φ, solve eq. (2.77) for θ_L.
    θ_L is assumed between 0 and φ (for 0 < φ < π).
    """
    # bracket for θ_L: small positive to slightly below φ
    eps = 1e-4
    theta_min = eps
    theta_max = max(phi - eps, theta_min + eps)

    def f(theta_L):
        return Lambda_from_thetaL(phi, theta_L) - Lambda_target

    sol = root_scalar(f, bracket=[theta_min, theta_max], xtol=tol)
    if not sol.converged:
        raise RuntimeError("solve_theta_L_for_Lambda did not converge")
    return sol.root


def elastica_centerline_analytical(Lambda_target, phi, L, n_pts=400):
    """
    Full analytical centerline (x(s), y(s)) based on eqs. (2.74)–(2.82).

    Returns:
        s_phys : arc-length coordinate [m], shape (n_pts,)
        x_phys: x-coordinate [m], shape (n_pts,)
        y_phys: y-coordinate [m], shape (n_pts,)
    """

    # 1) Solve for θ_L from Λ
    theta_L = solve_theta_L_for_Lambda(Lambda_target, phi)

    # 2) Build θ-grid from 0 to θ_L (avoid endpoint singularity)
    eps = 1e-6
    theta = np.linspace(0.0, theta_L - eps, n_pts)

    # 3) Compute integrand arrays
    w  = np.cos(phi - theta_L) - np.cos(phi - theta)
    f  = 1.0 / np.sqrt(w)         # for ξ
    fX = np.cos(theta) * f        # for X
    fY = np.sin(theta) * f        # for Y

    # 4) Cumulative integrals using trapezoidal rule
    xi_partial = cumulative_trapezoid(f,  theta, initial=0.0)  # ξ(φ, θ)
    X_partial  = cumulative_trapezoid(fX, theta, initial=0.0)  # X(φ, θ)
    Y_partial  = cumulative_trapezoid(fY, theta, initial=0.0)  # Y(φ, θ)

    xi_total = xi_partial[-1]   # ξ(φ, θ_L)

    # 5) Dimensionless arc length and coordinates
    #    s/L = ξ(φ,θ) / ξ(φ,θ_L)
    #    δx/L = X(φ,θ) / ξ(φ,θ_L)
    #    δy/L = Y(φ,θ) / ξ(φ,θ_L)
    s_over_L = xi_partial / xi_total
    x_over_L = X_partial  / xi_total
    y_over_L = Y_partial  / xi_total

    # 6) Convert to physical units
    s_phys = L * s_over_L
    x_phys = L * x_over_L
    y_phys = L * y_over_L

    return s_phys, x_phys, y_phys, theta_L

from scipy.integrate import cumulative_trapezoid

def centerline_from_theta(s_hat, theta, L):
    """
    Given θ(ŝ) and L, compute physical centerline (x(s), y(s)).
    ŝ in [0,1], s = L ŝ.
    """
    # Integrals in terms of ŝ:
    # x(ŝ) = L ∫_0^ŝ cos θ(ξ) dξ
    # y(ŝ) = L ∫_0^ŝ sin θ(ξ) dξ
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_hat = cumulative_trapezoid(cos_t, s_hat, initial=0.0)
    y_hat = cumulative_trapezoid(sin_t, s_hat, initial=0.0)

    x_phys = L * x_hat
    y_phys = L * y_hat

    s_phys = L * s_hat
    return s_phys, x_phys, y_phys
# Lambda = Lambda_from_B(0.04)
# print(Lambda)
geom = dde.geometry.Interval(0, 1)

plt.figure(figsize=(10, 5))

tip_phis   = []   # field directions
tip_pinn   = []   # tip angles from PINN
tip_ana    = []   # tip angles from analytic

for B_mag, phi_B in field_cases:
    # ----- 1) PINN -----
    s_hat_pinn, theta_pinn = solve_theta_for_field(B_mag, phi_B, geom, iters=8000)
    s_phys_pinn, x_pinn, y_pinn = centerline_from_theta(s_hat_pinn, theta_pinn, L_cat)
    theta_tip_pinn = theta_pinn[-1]

    # ----- 2) Analytic -----
    Lambda = Lambda_from_B(B_mag)
    if B_mag > 0.0:
        s_ana, x_ana, y_ana, theta_tip_ana = elastica_centerline_analytical(
            Lambda, phi_B, L_cat, n_pts=400
        )
    else:
        s_ana = np.linspace(0.0, L_cat, 400)
        x_ana = s_ana.copy()
        y_ana = np.zeros_like(s_ana)
        theta_tip_ana = 0.0

    # store for angle plot (use φ in degrees)
    tip_phis.append(np.degrees(phi_B))
    tip_pinn.append(theta_tip_pinn)
    tip_ana.append(theta_tip_ana)

    label_base = f"|B|={B_mag*1000:.1f} mT, φ={phi_B:.2f} rad"
    print(
        f"{label_base}: "
        f"θ_tip PINN = {theta_tip_pinn:.4f} rad ({np.degrees(theta_tip_pinn):.2f}°), "
        f"θ_tip analytic = {theta_tip_ana:.4f} rad ({np.degrees(theta_tip_ana):.2f}°)"
    )
    f_y_ana = interp1d(s_ana, y_ana, kind="cubic", fill_value="extrapolate")
    y_ana_on_pinn = f_y_ana(s_phys_pinn)

    l2_err = np.sqrt(np.mean((y_pinn - y_ana_on_pinn)**2))
    print(f"|B|={B_mag*1000:.1f} mT, φ={phi_B:.2f} rad: L2 error in y(s) = {l2_err:.3e} m")
    # centerline plot
    plt.plot(s_phys_pinn, y_pinn, "-",  label=label_base + " (PINN)")
    plt.plot(s_ana,  y_ana,  "--", label=label_base + " (analytic)")

plt.xlabel("x(s) [m]")
plt.ylabel("y(s) [m]")
plt.title("Magnetically actuated elastica: centerline & tip angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Tip angle vs field-direction plot ----------
plt.figure(figsize=(6, 4))
plt.plot(tip_phis, np.degrees(tip_pinn), "o-", label="PINN tip angle")
plt.plot(tip_phis, np.degrees(tip_ana),  "s--", label="Analytic tip angle")
plt.xlabel("Field direction φ [deg]")
plt.ylabel("Tip angle θ_L [deg]")
plt.title("Tip angle vs field direction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
