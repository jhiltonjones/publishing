import numpy as np
from beam_pcc import local_jacobians_from_gamma, pose_from_gamma, hat, stack_blocks
from numpy.polynomial.legendre import leggauss

def build_k(E, r, nu, lengths):
    E  = np.atleast_1d(E); nu = np.atleast_1d(nu)
    L  = np.atleast_1d(lengths); r = np.atleast_1d(r)
    if r.size == 1: r = np.full(len(L), r.item())
    K = np.zeros((3*len(L), 3*len(L)))
    for i in range(len(L)):
        I_i = (np.pi*r[i]**4)/4      
        J_i = 2.0 * I_i           
        G_i = E[i] / (2.0 * (1.0 + nu[i]))
        Ki  = np.diag([E[i]*I_i, E[i]*I_i, G_i*J_i])
        K[3*i:3*i+3, 3*i:3*i+3] = Ki
    return K


def G_matrix_build(axes, thetas, lengths, masses, R_base = None, p_base=None, quad_n=12):
    """
    This builds the G matrix in equation 16
    This is computed by M which is the mass density* matrix of stacked jacobians * g
    """

    n = len(lengths)
    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    S = np.zeros((3*n, 3))

    for i, (n_hat, theta, L) in enumerate(zip(axes, thetas, lengths)):
        gamma_i = (theta/L)*np.asarray(n_hat)
        xi, wi = leggauss(quad_n)
        si = 0.5*(xi+1)*L
        w = 0.5 * L*wi

        Int = np.zeros((3,3))

        for s_k, w_k in zip(si,w):
            Jp,_ = local_jacobians_from_gamma(gamma_i, s_k, quad_n=max(6,quad_n//2))
            Jp = R_up @ Jp
            Int += w_k * Jp.T
        S[3*i:3*i+3, :] = Int

        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)

    M = np.zeros((3*n, 3*n))
    for i, m in enumerate(masses):
        M[3*i: 3*i+3, 3*i:3*i+3] = m * np.eye(3)
    def apply_to_g(g):
        g = np.asarray(g).reshape(3)
        return M@(S@g)  
    return apply_to_g, S 
def energy_and_grad(axes, thetas, lengths, masses, gvec, E, r, nu,
                    R_base=None, p_base=None, quad_n=12):
    """
    This codes equation 13 and equation 15. The U grad codes the second part in equation 13
    """
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])

    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))
    U_stiff = 0.5 * gamma @ (K @ gamma)

    U_grav = 0.0
    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()
    for n_hat, theta, L, m in zip(axes, thetas, lengths, masses):
        gamma_i = (theta/L) * np.asarray(n_hat,float)
        xi, wi = leggauss(quad_n)
        si = 0.5*(xi+1.0)*L
        w  = 0.5*L*wi
        for s, ws in zip(si, w):
            _, p_s = pose_from_gamma(R_up, p_up, gamma_i, s)
            U_grav += - m * (gvec @ p_s) * ws
        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)

    apply_G, _S = G_matrix_build(axes, thetas, lengths, masses, R_base, p_base, quad_n)
    Qg = apply_G(gvec)     

    grad = (K @ gamma) - Qg

    U = U_stiff + U_grav
    return U, grad, K

def H_matrix_build(axes, thetas, lengths, eta0_list,
                   R_base=None, p_base=None, quad_n=12):
    """
    Build H(γ) so that τ = H(γ) B + J(γ)^T f_e  (Eq. 19).

    eta0_list: list of 3-vectors η_{j0} (magnetic dipole per unit length) in each segment's local frame.
    """
    n = len(lengths)
    H = np.zeros((3*n, 3))

    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    for j, (n_hat, theta, L, eta0) in enumerate(zip(axes, thetas, lengths, eta0_list)):
        gamma_j = (theta / L) * np.asarray(n_hat, dtype=float)

        xi, wi = leggauss(quad_n)
        si = 0.5 * (xi + 1.0) * L
        ws = 0.5 * L * wi

        Hj = np.zeros((3, 3))
        for s_k, w_k in zip(si, ws):
            R_s, _ = pose_from_gamma(R_up, p_up, gamma_j, s_k)
            _, Jo_loc = local_jacobians_from_gamma(gamma_j, s_k, quad_n=max(6, quad_n//2))
            Jo_world = R_s @ Jo_loc   
            eta_s = R_s @ np.asarray(eta0)
            Hj += w_k * (Jo_world.T @ hat(eta_s))
        H[3*j:3*j+3, :] = Hj
        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_j, L)

    return H

def tau_from_eq19(axes, thetas, lengths, eta0_list, B, f_e,
                  R_base=None, p_base=None, quad_n=12):
    """
    Compute τ = H(γ) B + J(γ)^T f_e   exactly as Eq. (19).

    - B: 3-vector external magnetic field in world frame.
    - f_e: stacked 6n-vector of concentrated wrenches at each segment end
           [f1; m1; f2; m2; ...] in world frame.
    """
    H = H_matrix_build(axes, thetas, lengths, eta0_list, R_base, p_base, quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    tau = H @ np.asarray(B).reshape(3) + J.T @ np.asarray(f_e).reshape(-1)
    return tau, H, J

def eq20_residual(axes, thetas, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                  R_base=None, p_base=None, quad_n=12):
    """
    Residual of Eq. (20): r = [-Kγ - G(γ)] - [H(γ)B + J(γ)^T f_e]
    Returns (r, pieces...) so you can debug/inspect.
    """
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat, float)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])
    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))
    Kgamma = K @ gamma
    apply_G, _S = G_matrix_build(axes, thetas, lengths, masses,
                                 R_base=R_base, p_base=p_base, quad_n=quad_n)
    Qg = apply_G(gvec)  
    H = H_matrix_build(axes, thetas, lengths, eta0_list,
                       R_base=R_base, p_base=p_base, quad_n=quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    right = H @ np.asarray(B).reshape(3) + J.T @ np.asarray(f_e).reshape(-1)
    left  = -Kgamma - Qg
    r = left - right
    return r, left, right, K, H, J, Qg, gamma


def central_difference_jacobian(func, x, rel_eps=1e-6, abs_eps=1e-10, h_floor=1e-4, *args, **kwargs):
    x = np.asarray(x, float)
    n = x.size
    r0 = np.asarray(func(x, *args, **kwargs), float)
    m = r0.size
    J = np.zeros((m, n))
    for j in range(n):
        h = max(abs_eps, rel_eps * max(1.0, abs(x[j])), h_floor)
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        rp = np.asarray(func(xp, *args, **kwargs), float)
        rm = np.asarray(func(xm, *args, **kwargs), float)
        J[:, j] = (rp - rm) / (2.0 * h)
    return r0, J

def residual_theta(thetas, axes, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                   R_base=None, p_base=None, quad_n=12):
    r, *_ = eq20_residual(axes, thetas, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
                          R_base=R_base, p_base=p_base, quad_n=quad_n)
    return r
# def guass_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e, gvec, E, r, nu,
#                     R_base=None, p_base=None, quad_n=12,
#                     max_iter=50, tol_r=1e-8, tol_step=1e-8,
#                     lambda_init=1e-8, lambda_up=10.0, lambda_down=0.1,
#                     h_floor=5e-3):
#     theta = np.asarray(thetas0, float).copy()
#     lam = float(lambda_init)

#     for k in range(max_iter):
#         res, J = central_difference_jacobian(
#             residual_theta, theta,
#             rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
#             axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
#             B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
#             R_base=R_base, p_base=p_base, quad_n=quad_n
#         )
#         res_norm = np.linalg.norm(res)
#         if res_norm < tol_r:
#             break

#         JTJ  = J.T @ J
#         grad = J.T @ res                         
#         jtj_scale = float(np.max(np.diag(JTJ)))
#         jtj_scale = max(jtj_scale, 1e-20)
#         A_sys = JTJ + (lam * jtj_scale) * np.eye(JTJ.shape[0])

#         try:
#             step = -np.linalg.solve(A_sys, grad)
#         except np.linalg.LinAlgError:
#             step = -np.linalg.pinv(A_sys) @ grad

#         if np.linalg.norm(step, np.inf) < tol_step:
#             break

#         max_step = 0.5
#         step = np.clip(step, -max_step, max_step)

#         t = 1.0
#         while t > 1e-6:
#             res_try = residual_theta(theta + t*step, axes, lengths, masses, eta0_list,
#                                      B, f_e, gvec, E, r, nu,   # <- pass gvec and r unchanged
#                                      R_base=R_base, p_base=p_base, quad_n=quad_n)
#             if np.linalg.norm(res_try) < res_norm:
#                 theta = theta + t*step
#                 lam   = max(lam * lambda_down, 1e-12)
#                 break
#             t *= 0.5

#         if t <= 1e-6:
#             lam *= lambda_up

#     return theta
def gauss_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e,
                          gvec, E, r, nu,
                          R_base=None, p_base=None, quad_n=12,
                          max_iter=50,
                          tol_grad=1e-8,      # NEW: gradient stopping
                          tol_step=1e-10,     # relative step stopping
                          tol_r=1e-8,
                          lambda_init=1e-3,   # typical LM starting value
                          h_floor=5e-3):
    """
    Full Gavin-style Levenberg-Marquardt with rho update and no line search.
    """

    theta = np.asarray(thetas0, float).copy()
    lam = float(lambda_init)

    for k in range(max_iter):

        # ---- Evaluate residual and Jacobian ---------
        res, J = central_difference_jacobian(
            residual_theta, theta,
            rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
            axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
            B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )

        res_norm = np.linalg.norm(res)
        if res_norm < tol_r:
            break

        # ---- First-order optimality test ----
        grad = J.T @ res
        if np.linalg.norm(grad, np.inf) < tol_grad:
            break

        # ---- Build LM system ----------
        JTJ = J.T @ J
        scale = float(np.max(np.diag(JTJ)))
        scale = max(scale, 1e-12)

        A = JTJ + lam * scale * np.eye(len(theta))

        try:
            h = -np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(A) @ grad

        # ---- Relative step test ----
        if np.linalg.norm(h) < tol_step * (np.linalg.norm(theta) + 1e-12):
            break

        # ---- Compute predicted reduction (Gavin Eq. 15/16) ----
        # pred = | hᵀ ( λ h + Jᵀ r ) |
        pred = np.abs(h @ (lam * scale * h + grad))

        if pred < 1e-30:   # step too small to be meaningful
            break

        # ---- Evaluate new residual ----
        theta_new = theta + h
        res_new = residual_theta(theta_new, axes, lengths, masses, eta0_list,
                                 B, f_e, gvec, E, r, nu,
                                 R_base=R_base, p_base=p_base, quad_n=quad_n)

        res_new_norm = np.linalg.norm(res_new)

        # ---- Actual reduction ----
        act = 0.5*(res_norm**2 - res_new_norm**2)

        # ---- Compute rho (Eq. 14) ----
        rho = act / pred

        # ---- Update step depending on rho (Gavin Sec 4.1.1 method 3) ----
        if rho > 0:      # success → accept step
            theta = theta_new
            res = res_new
            res_norm = res_new_norm

            # decrease lambda (more Gauss-Newton-like)
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            lam = max(lam, 1e-15)

        else:            # failure → reject step
            # increase lambda (more gradient-descent-like)
            lam = lam * min(10, (1/(1+rho)))
            continue

    return theta

def magnetic_field_for_theta(thetas, axes, lengths, masses, eta0_list,
                             f_e, gvec, E, r, nu,
                             R_base=None, p_base=None, quad_n=12,
                             prefer_dir=None,  # e.g. np.array([0,0,1]) to force B along z
                             reg=0.0):         # Tikhonov (ridge) regularization
    """
    Solve Eq. (20) for B given thetas (static equilibrium):
        [-Kγ - G(γ)] = H(γ) B + J(γ)^T f_e
      => H B = -Kγ - Qg - J^T f_e  (linear in B)

    If prefer_dir is provided (3-vector), solve only for magnitude along that direction.
    """
    thetas = np.asarray(thetas, float)
    # gamma from thetas
    gamma = np.concatenate([(theta/L)*np.asarray(n_hat, float)
                            for n_hat, theta, L in zip(axes, thetas, lengths)])

    # Elastic term
    K = build_k(np.asarray(E), np.asarray(r), np.asarray(nu), np.asarray(lengths))

    # Gravity generalized load Qg
    apply_G, _ = G_matrix_build(axes, thetas, lengths, masses,
                                R_base=R_base, p_base=p_base, quad_n=quad_n)
    Qg = apply_G(gvec)

    # Magnetic mapping H and Jacobian J for concentrated wrenches
    H = H_matrix_build(axes, thetas, lengths, eta0_list,
                       R_base=R_base, p_base=p_base, quad_n=quad_n)
    _, _, J, _, _ = stack_blocks(axes, thetas, lengths, s_list=lengths,
                                 R_base=R_base, p_base=p_base)

    # Right-hand side for H B = rhs
    rhs = -(K @ gamma) - Qg - (J.T @ np.asarray(f_e).reshape(-1))

    # Solve for B
    # if prefer_dir is not None:
    #     # Constrain B = alpha * u
    #     u = np.asarray(prefer_dir, float).reshape(3)
    #     nu = np.linalg.norm(u)
    #     if nu == 0:
    #         raise ValueError("prefer_dir must be nonzero")
    #     u = u / nu
    #     Hu = H @ u
    #     denom = (Hu @ Hu) + float(reg)
    #     if denom < 1e-16:
    #         raise np.linalg.LinAlgError("Direction leads to near-singular system.")
    #     alpha = (Hu @ rhs) / denom
    #     B = alpha * u
    # else:
        # Unconstrained least squares (with optional ridge)
        # Solve (H^T H + reg I) B = H^T rhs
    HtH = H.T @ H
    if reg > 0:
        HtH = HtH + reg * np.eye(3)
    try:
        B = np.linalg.solve(HtH, H.T @ rhs)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(H) @ rhs  # fallback

    # Diagnostics
    res = H @ B - rhs
    info = {
        "rhs_norm": float(np.linalg.norm(rhs)),
        "residual_norm": float(np.linalg.norm(res)),
        "H_cond_est": float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else np.inf,
        "B_T": B
    }
    return B, info
def my_gauss_newton_lm(thetas0, axes, lengths, masses, eta0_list, B, f_e,
                       gvec, E, r, nu,
                       R_base=None, p_base=None, quad_n=12,
                       max_iter=80,
                       tol_grad=1e-10,
                       tol_step=1e-12,
                       tol_r=1e-10,
                       lambda_init=1e-4,
                       h_floor=1e-5):
    theta = np.asarray(thetas0, float).copy()
    lam = float(lambda_init)

    for _ in range(max_iter):
        # residual and Jacobian of *residual_theta*, not eq20_residual
        res, J = central_difference_jacobian(
            residual_theta, theta,
            rel_eps=1e-6, abs_eps=1e-10, h_floor=h_floor,
            axes=axes, lengths=lengths, masses=masses, eta0_list=eta0_list,
            B=B, f_e=f_e, gvec=gvec, E=E, r=r, nu=nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_norm = np.linalg.norm(res)
        if res_norm < tol_r:
            break

        grad = J.T @ res
        if np.linalg.norm(grad, np.inf) < tol_grad:
            break

        JTJ = J.T @ J
        scale = max(np.max(np.diag(JTJ)), 1e-12)
        A = JTJ + lam * scale * np.eye(len(theta))

        try:
            h = -np.linalg.solve(A, grad)
        except np.linalg.LinAlgError:
            h = -np.linalg.pinv(A) @ grad

        if np.linalg.norm(h) < tol_step * (np.linalg.norm(theta) + 1e-12):
            break

        theta_new = theta + h
        # evaluate new residual
        res_new = residual_theta(
            theta_new, axes, lengths, masses, eta0_list,
            B, f_e, gvec, E, r, nu,
            R_base=R_base, p_base=p_base, quad_n=quad_n
        )
        res_new_norm = np.linalg.norm(res_new)

        act = 0.5 * (res_norm**2 - res_new_norm**2)
        pred = np.abs(h @ (lam * scale * h + grad))
        rho = act / (pred + 1e-30)

        if rho > 0:
            theta = theta_new
            res = res_new
            res_norm = res_new_norm
            lam = lam * max(1/3, 1 - (2*rho - 1)**3)
            lam = max(lam, 1e-15)
        else:
            lam = lam * min(10, 1/(1+rho))

    return theta

