import numpy as np
import math

# --- Existing helper: rotation vector -> rotation matrix (Rodrigues) ---
def rotvec_to_rotmat(r):
    r = np.array(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

# --- Your function: rotation matrix -> rotation vector (axis * angle) ---
def rot_to_axis_angle(R):
    eps = 1e-12
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # clamp for numerical stability
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)  # rotation vector

def apply_z_rotation_deg_to_tcp(tcp_pose, delta_deg):
    """
    tcp_pose: [x, y, z, rx, ry, rz]  (UR-style axis-angle)
    delta_deg: rotation about *base* Z axis in degrees (right-hand rule)

    Returns a new TCP pose [x, y, z, rx', ry', rz'].
    """
    tcp_pose = list(tcp_pose)
    pos = np.array(tcp_pose[0:3], dtype=float)
    r   = np.array(tcp_pose[3:6], dtype=float)

    # current orientation as matrix
    R_current = rotvec_to_rotmat(r)

    # rotation about base Z axis
    theta = math.radians(delta_deg)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # New orientation: first rotate around base Z, then apply old orientation
    R_new = Rz @ R_current

    # back to rotation vector
    r_new = np.array(rot_to_axis_angle(R_new))

    return np.concatenate([pos, r_new])

# -----------------------------------------------------------------
# Example with your pose:
tcp = [
0.5050571191387655, -0.33238898342243606, 0.8108449509242835, -1.8010614846248143, 2.5399321431490725, 0.07943768069463192
]

tcp_new = apply_z_rotation_deg_to_tcp(tcp, 30.0)

print("Original TCP:", tcp)
print("New TCP (+30° about base Z):", tcp_new.tolist())
import numpy as np

theta = np.deg2rad(10.0)
c, s = np.cos(theta), np.sin(theta)
# 3D Rotation Matrix around the X-axis
R_x = np.array([
    [1.0, 0.0, 0.0],
    [0.0,   c,  -s],
    [0.0,   s,   c]
])

print(f"Rotation matrix R_x for {np.rad2deg(theta)} degrees:")
print(R_x)
R_epm2 = np.array([
    [c, -s, 0.0],
    [s,  c, 0.0],
    [0.0, 0.0, 1.0]
])
R_epm =np.array([
    [ -1.0,          -0.0012,         0],
    [0.0012, -1,0],
    [0,  0.0, 1] 
  ])

R_y = np.array([
    [ c, 0.0,  s],
    [0.0, 1.0, 0.0],
    [-s, 0.0,  c]
])

print(f"Rotation matrix R_y for {np.rad2deg(theta)} degrees:")
print(R_y)
print(R_epm)
tcp_pose = list(tcp)
pos = np.array(tcp_pose[0:3], dtype=float)
r   = np.array(tcp_pose[3:6], dtype=float)
R_current = rotvec_to_rotmat(r)
R_new = R_x @ R_current
rx, ry, rz = rot_to_axis_angle(R_new)  # radians

x, y, z = tcp[:3]   # wherever you want the centre to be, e.g. current TCP

TCP_TARGET = [float(x), float(y), float(z),
              float(rx), float(ry), float(rz)]
print(TCP_TARGET)

