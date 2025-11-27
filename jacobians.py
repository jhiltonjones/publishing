import numpy as np
from run_solver import *
def forward_tip_pose(axes, thetas, lengths, R_base=None, p_base=None):
    """
    Integrate along all segments and return final (R_tip, p_tip).
    """
    R = np.eye(3) if R_base is None else R_base.copy()
    p = np.zeros(3) if p_base is None else p_base.copy()

    for n_hat, theta, L in zip(axes, thetas, lengths):
        gamma_i = (theta / L) * np.asarray(n_hat, float)
        R, p = pose_from_gamma(R, p, gamma_i, L)

    return R, p

def so3_log(R):
    """
    Log map from SO(3) to so(3) (axis-angle vector).
    """
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        # Very small angle: use first-order approximation
        return 0.5 * np.array([R[2,1] - R[1,2],
                               R[0,2] - R[2,0],
                               R[1,0] - R[0,1]])

    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2.0 * np.sin(theta))
    return theta * w
def tip_pose_6d_from_thetas(thetas, axes, lengths, R_base=None, p_base=None):
    """
    Returns a 6D pose vector [p; phi] where:
      p   = tip position (3,)
      phi = axis-angle of tip orientation (3,)
    """
    R_tip, p_tip = forward_tip_pose(axes, thetas, lengths, R_base, p_base)
    phi = so3_log(R_tip)              # orientation vector
    return np.hstack([p_tip, phi])    # shape (6,)
# thetas0 is your current guess/solution; same axes, lengths as above

pose0, J_theta = central_difference_jacobian(
    tip_pose_6d_from_thetas,
    thetas0,
    axes=axes, lengths=lengths,
    R_base=None, p_base=None,   # or your actual base pose
    rel_eps=1e-6, abs_eps=1e-10, h_floor=1e-4
)

print("Tip pose at thetas0:", pose0)         # [px, py, pz, φx, φy, φz]
print("Jacobian d pose / d theta shape:", J_theta.shape)   # (6, n)
def tip_pose_6d_from_theta_and_L(x, axes, R_base=None, p_base=None):
    """
    x contains [thetas, lengths] concatenated.
    """
    n = len(axes)
    thetas  = x[:n]
    lengths = x[n:]
    return tip_pose_6d_from_thetas(thetas, axes, lengths, R_base, p_base)
x0 = np.hstack([thetas0, lengths])   # current angles and current lengths

pose0, J_full = central_difference_jacobian(
    tip_pose_6d_from_theta_and_L,
    x0,
    axes=axes,
    R_base=None, p_base=None,
    rel_eps=1e-6, abs_eps=1e-10, h_floor=1e-4
)

n = len(axes)
J_theta  = J_full[:, :n]     # ∂pose/∂theta
J_L      = J_full[:, n:]     # ∂pose/∂length
