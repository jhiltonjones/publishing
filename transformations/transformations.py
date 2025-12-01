import numpy as np

def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])

def rotx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def transGen(R, t):
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)"""
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def rotvec_to_R(r):
    r = np.asarray(r)
    theta = np.linalg.norm(r)
    if theta < 1e-9:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array([
        [0,    -kz,   ky],
        [kz,    0,   -kx],
        [-ky,  kx,    0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def R_to_rotvec(R):
    R = np.asarray(R)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    k = np.array([rx, ry, rz])
    return theta * k



def rotate_around_point_transform(axis, pivot_pos, theta):

    if axis in ('x', 1):
        R = rotx(theta)
    elif axis in ('y', 2):
        R = roty(theta)
    else:
        R = rotz(theta)

    c = np.asarray(pivot_pos).reshape(3) 
    I = np.eye(3)
    t = (I - R) @ c                 
    return transGen(R, t)


start_point = np.array([
    0.8591984247948907,
   -0.2714764828605954,
    0.20603768069158077,
    2.3911077741703193,
   -1.9880074589982653,
    0.02729820295870479
])

pivot_point = np.array([
   -0.107215706502096,
   -2.141261716882223,
   -1.714442491531372,
   -0.8591966790011902,
    1.5362006425857544,
   -0.2898953596698206
])


ee_pos0 = start_point[:3]
ee_rvec0 = start_point[3:]
R_b_e0 = rotvec_to_R(ee_rvec0)
H_b_e0 = transGen(R_b_e0, ee_pos0)


mag_offset = np.array([0, 0, 0.25])
H_e_m = transGen(np.eye(3), mag_offset)   


H_b_m0 = H_b_e0 @ H_e_m


pivot_pos = pivot_point[:3]


theta = np.deg2rad(90)
H_rot = rotate_around_point_transform('z', pivot_pos, theta)


H_b_m1 = H_rot @ H_b_m0

H_m_e = np.linalg.inv(H_e_m)
H_b_e1 = H_b_m1 @ H_m_e

new_pos = H_b_e1[:3, 3]
new_R   = H_b_e1[:3, :3]
new_rvec = R_to_rotvec(new_R)

new_pose_for_robot = np.hstack([new_pos, new_rvec])

print("New EE pose to send to robot:")
print(new_pose_for_robot)
