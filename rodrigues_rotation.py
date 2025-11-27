#computes the z axis rotation using the rodrigues formula, there are 3 methods which are the rotation of the z axis around an axis
# a matrix which creates a rotation matrix from the rx,ry,rz and a matrix which extracts the rx,ry,r4z values from this matrix

import math
import numpy as np 

def axis_angle_to_rot(rx, ry, rz):
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = rx/theta, ry/theta, rz/theta
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.eye(3) + math.sin(theta)*K + (1.0-math.cos(theta))*(K@K)
    return R

def rot_to_axis_angle(R):
    tr = np.trace(R)
    c = (tr - 1.0)/2.0
    c = max(min(c,1.0),-1.0)
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0,0.0,0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)

def Rz(yaw_rad):
    c,s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c,-s,0], [s,c,0], [0,0,1]])
