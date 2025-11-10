import numpy as np
from numpy.polynomial.legendre import leggauss

def hat(v):
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def so3_exp(omega):
    th=np.linalg.norm(omega); W=hat(omega)
    if th<1e-12: return np.eye(3)+W
    W2=W@W; s,c=np.sin(th),np.cos(th)
    return np.eye(3)+(s/th)*W+((1-c)/(th**2))*W2
def so3_left_jacobian(omega):
    th=np.linalg.norm(omega); W=hat(omega)
    if th<1e-12: return np.eye(3)+0.5*W+(1/6)*(W@W)
    W2=W@W; s,c=np.sin(th),np.cos(th)
    return np.eye(3)+((1-c)/(th**2))*W+((th-s)/(th**3))*W2
def so3_right_jacobian(omega):
    return so3_left_jacobian(-np.asarray(omega,float))
def pose_from_gamma(R0,p0,gamma,s):
    R=R0@so3_exp(gamma*s)
    e3=np.array([0.,0.,1.]); p=p0+R0@(s*so3_left_jacobian(gamma*s)@e3)
    return R,p

def local_jacobians_from_gamma(gamma, s, quad_n=16):
    Jo_local = s * so3_right_jacobian(gamma * s)
    e3 = np.array([0.0, 0.0, 1.0])
    Jp_local = np.zeros((3, 3))
    xi, wi = leggauss(quad_n)
    ui = 0.5 * (xi + 1.0) * s
    w_scaled = 0.5 * s * wi
    for j in range(3):
        ej = np.eye(3)[:, j]
        col = np.zeros(3)
        for u, w in zip(ui, w_scaled):
            R_rel = so3_exp(gamma * u)
            Jr = so3_right_jacobian(gamma * u)
            v = Jr @ (u * ej)
            col += w * (R_rel @ (hat(v) @ e3))
        Jp_local[:, j] = col
    return Jp_local, Jo_local
def stack_blocks(axes, thetas, lengths, s_list=None, R_base=None, p_base=None):

    n = len(lengths)
    if s_list is None:
        s_list = lengths

    R_up = np.eye(3) if R_base is None else R_base.copy()
    p_up = np.zeros(3) if p_base is None else p_base.copy()

    R_nodes = [R_up.copy()]
    p_nodes = [p_up.copy()]
    Rs_si, ps_si = [], []
    J = np.zeros((6*n, 3*n))

    for i, (n_hat, theta, L, s_i) in enumerate(zip(axes, thetas, lengths, s_list)):
        gamma_i = (theta / L) * np.asarray(n_hat) 

        Jp_loc, Jo_loc = local_jacobians_from_gamma(gamma_i, s_i)

        R_si, p_si = pose_from_gamma(R_up, p_up, gamma_i, s_i)
        Rs_si.append(R_si)
        ps_si.append(p_si)
        Jp_world = R_up @ Jp_loc
        Jo_world = R_si @ Jo_loc

        J_block = np.vstack([Jp_world, Jo_world])
        J[6*i:6*i+6, 3*i:3*i+3] = J_block

        R_up, p_up = pose_from_gamma(R_up, p_up, gamma_i, L)
        R_nodes.append(R_up.copy())
        p_nodes.append(p_up.copy())

    return np.array(Rs_si), np.array(ps_si), J, np.array(R_nodes), np.array(p_nodes)

