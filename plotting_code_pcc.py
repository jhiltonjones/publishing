from beam_pcc import *

axes    = [np.array([1.,0,0]), np.array([1.,0,0])]
thetas  = [np.deg2rad(-90), np.deg2rad(90)]
lengths = [0.8, 1.0]

Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(axes, thetas, lengths)

print("Pose at s_i inside each section:")
print(" p_1(s1) =", ps_si[0])
print(" p_2(s2) =", ps_si[1])

print("\nJ shape:", J.shape)     # (6n, 3n) = (12, 6)
print("Top-left 6x3 block J1:\n", J[:6, :3])
print("Bottom-right 6x3 block J2:\n", J[6:, 3:])
# --- Additional code for visualization only ---
# Creates a 3D plot of the backbone and shows Jacobian columns as arrows at s_i.
import numpy as np
import matplotlib.pyplot as plt

# If user variables aren't present, set a tiny demo so this cell runs standalone.
if "axes" not in globals():
    axes    = [np.array([1.,0,0]), np.array([0.,1,0])]
    thetas  = [np.deg2rad(30), np.deg2rad(-20)]
    lengths = [0.8, 1.0]

# Recompute stacked blocks (poses and Jacobian) if missing
if "stack_blocks" in globals():
    Rs_si, ps_si, J, R_nodes, p_nodes = stack_blocks(axes, thetas, lengths)
else:
    raise RuntimeError("Please run your model code first so stack_blocks is defined.")

# Build dense backbone samples for a smooth curve
pts = [p_nodes[0]]
R_up = R_nodes[0]
p_up = p_nodes[0]
for n_hat, th, L in zip(axes, thetas, lengths):
    gamma = (th/L) * np.asarray(n_hat, float)
    us = np.linspace(0, L, 60)
    for u in us[1:]:
        R_u, p_u = pose_from_gamma(R_up, p_up, gamma, u)
        pts.append(p_u)
    # advance to end of section
    R_up, p_up = pose_from_gamma(R_up, p_up, gamma, L)

pts = np.vstack(pts)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2)

# Show nodes
ax.scatter(p_nodes[:,0], p_nodes[:,1], p_nodes[:,2], s=30)

# Draw linear Jacobian columns at s_i as small arrows
scale = 0.5
n = len(axes)
for i in range(n):
    # world-frame 6x3 block for section i
    # Top 3 rows = linear, bottom 3 rows = angular
    J_block = J[6*i:6*i+6, 3*i:3*i+3]
    Jp_world = J_block[:3, :]
    p_i = ps_si[i]
    # three little arrows for the 3 columns of Jp
    for j in range(3):
        d = Jp_world[:, j] * scale
        ax.quiver(p_i[0], p_i[1], p_i[2], d[0], d[1], d[2], length=1.0, normalize=False)

# Label and equal-ish aspect
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
mins = pts.min(axis=0); maxs = pts.max(axis=0)
cent = (mins+maxs)/2.0; span = (maxs-mins).max()
ax.set_xlim([cent[0]-span/2, cent[0]+span/2])
ax.set_ylim([cent[1]-span/2, cent[1]+span/2])
ax.set_zlim([cent[2]-span/2, cent[2]+span/2])

plt.show()
