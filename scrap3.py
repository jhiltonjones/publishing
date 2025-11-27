import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from run_solver import magnetic_field_for_theta, single_epm_pose_from_B, B_from_dipole
from energy_minimisation import my_gauss_newton_lm, energy_and_grad
# =========================
# 1. GEOMETRY & MATERIAL
# =========================

L_cat = 0.0400   # total catheter length [m]
L1    = 0.04     # single-segment length [m]

# external dipole magnitude for EPM pose solver (A·m²)
mu_mag = 283.0

# direction from workspace to magnet (for single_epm_pose_from_B)
e_r   = np.array([1.0, 0.0, 0.0])

# catheter radius (you can change this if needed)
r1 = 0.0005   # [m]  (0.1 mm)

# external permanent magnet position used in bending simulation
# (x, y, z) in meters – here 7 cm horizontally, 4 cm vertically
r_epm = np.array([0.13, 0.04, 0.0])
r_epm2 = np.array([0.12, 0.04, 0.0])
r_epm3 = np.array([0.11, 0.04, 0.0])

# NiTi material
E_niti  = 2.8e6        # Young's modulus [Pa]
rho_niti = 6700       # density [kg/m³]
nu_niti  = 0.33       # Poisson's ratio (for mechanics, may override in arrays)
gvec     = np.array([0.0, 0.0, -9.8])

# Cross-sectional properties
A_cs = np.pi * r1**2
I1   = np.pi * r1**4 / 4.0
EI1  = E_niti * I1

# Mass
m1 = rho_niti * A_cs * L_cat

# Beam axes and segment data
axes    = [np.array([-1.0, 0.0, 0.0], float)]  # bending axis
lengths = np.array([L1], dtype=float)
n       = len(lengths)

# Section arrays (1 segment)
r_arr  = np.array([r1],       dtype=float)
E_arr  = np.array([E_niti],   dtype=float)
nu_arr = np.array([0.49],     dtype=float)  # nearly incompressible

# Mass per unit length and segment masses
lam    = rho_niti * A_cs
masses = lam * lengths

# 6n external wrench vector
f_e = np.zeros(6*n, dtype=float)

# Magnetic linear density along inner IPM
Br  = 1.4
mu0 = 4e-7 * np.pi
D_ipm = 3e-3
L_ipm = 4e-2
V_ipm = np.pi * (0.5 * D_ipm)**2 * L_ipm
mu_ipm_mag = Br * V_ipm / mu0
mu_ipm_hat = np.array([0.0, 0.0, 1.0])
mu_ipm = mu_ipm_mag * mu_ipm_hat

m_dir_local = np.array([0.0, 0.0, -1.0], float)
eta_mag     = (mu_ipm_mag / L_ipm) * m_dir_local
eta0_list   = [eta_mag.copy()]
print(f"The magnetisation is {eta_mag}")
# initial guess for segment angles
thetas0 = np.zeros(n, float)
D_epm = 80e-3
L_epm = 90e-3
V_epm = np.pi*(0.5*D_epm)**2*L_epm
mu_epm_mag = Br*V_epm/mu0
mu_epm_hat = np.array([0.0, 0,1])
mu_epm = mu_epm_mag * mu_epm_hat
# ================================
# 2. BENDING VS MAGNET ROTATION
# ================================

# target small bending angle at 0° for calibration (2 deg)
thetas_target = np.array([np.deg2rad(2.0)])

B_req, info = magnetic_field_for_theta(
    thetas_target, axes, lengths, masses, eta0_list,
    f_e=np.zeros(6*n),
    gvec=gvec,
    E=r_arr*0 + E_arr,
    r=r_arr,
    nu=nu_arr
)

# Find one EPM pose that generates this field at origin
r_vec, mu_hat, R_epm0 = single_epm_pose_from_B(B_req, mu_mag, e_r)

print("Calibration:")
print("  B_req (T):", B_req)
print("  EPM position r_vec (m):", r_vec, "|r| =", np.linalg.norm(r_vec))
print("  μ̂:", mu_hat)
print("  R_epm0:\n", R_epm0)

phis_deg = np.linspace(0.0, 180.0, 361)   # 0.5° steps
phis_rad = np.deg2rad(phis_deg)
thetas_all = np.zeros((len(phis_rad), 1))
thetas_init = thetas0.copy()
n_seg = len(lengths)
thetas_all = np.zeros((len(phis_rad), n_seg))  # store bending per segment


for i, phi in enumerate(phis_rad):
    R_epm_phi  = np.array([
        [1.0,         0.0,          0.0],
        [0.0,  np.cos(phi), -np.sin(phi)],
        [0.0,  np.sin(phi),  np.cos(phi)]
    ]) @ R_epm0

    mu_hat_phi = R_epm_phi[:, 2]
    B_phi      = B_from_dipole(np.zeros(3), r_epm, mu_hat_phi, mu_mag)

    thetas_phi = my_gauss_newton_lm(
        thetas_init,
        axes, lengths, masses, eta0_list,
        B=B_phi,
        f_e=np.zeros(6),
        gvec=gvec,
        E=E_arr,
        r=r_arr,
        nu=nu_arr
    )


    thetas_all[i, :] = thetas_phi
    thetas_init = thetas_phi.copy()

thetas_all_deg = np.rad2deg(thetas_all[:, 0])

# convert to degrees
sim_deg = np.rad2deg(thetas_all[:, 0])




phis_deg = np.linspace(0.0, 180.0, 361)   # 0.5° steps
phis_rad = np.deg2rad(phis_deg)
thetas_all2 = np.zeros((len(phis_rad), 1))
thetas_init = thetas0.copy()
n_seg = len(lengths)
thetas_all2 = np.zeros((len(phis_rad), n_seg))  # store bending per segment


for i, phi in enumerate(phis_rad):
    R_epm_phi  = np.array([
        [1.0,         0.0,          0.0],
        [0.0,  np.cos(phi), -np.sin(phi)],
        [0.0,  np.sin(phi),  np.cos(phi)]
    ]) @ R_epm0

    mu_hat_phi = R_epm_phi[:, 2]
    B_phi      = B_from_dipole(np.zeros(3), r_epm2, mu_hat_phi, mu_mag)

    thetas_phi = my_gauss_newton_lm(
        thetas_init,
        axes, lengths, masses, eta0_list,
        B=B_phi,
        f_e=np.zeros(6),
        gvec=gvec,
        E=E_arr,
        r=r_arr,
        nu=nu_arr
    )


    thetas_all2[i, :] = thetas_phi
    thetas_init = thetas_phi.copy()

thetas_all_deg2 = np.rad2deg(thetas_all2[:, 0])

# convert to degrees
sim_deg2 = np.rad2deg(thetas_all2[:, 0])



phis_deg = np.linspace(0.0, 180.0, 361)   # 0.5° steps
phis_rad = np.deg2rad(phis_deg)
thetas_all3 = np.zeros((len(phis_rad), 1))
thetas_init = thetas0.copy()
n_seg = len(lengths)
thetas_all3 = np.zeros((len(phis_rad), n_seg))  # store bending per segment


for i, phi in enumerate(phis_rad):
    R_epm_phi  = np.array([
        [1.0,         0.0,          0.0],
        [0.0,  np.cos(phi), -np.sin(phi)],
        [0.0,  np.sin(phi),  np.cos(phi)]
    ]) @ R_epm0

    mu_hat_phi = R_epm_phi[:, 2]
    B_phi      = B_from_dipole(np.zeros(3), r_epm3, mu_hat_phi, mu_mag)

    thetas_phi3 = my_gauss_newton_lm(
        thetas_init,
        axes, lengths, masses, eta0_list,
        B=B_phi,
        f_e=np.zeros(6),
        gvec=gvec,
        E=E_arr,
        r=r_arr,
        nu=nu_arr
    )


    thetas_all3[i, :] = thetas_phi3
    thetas_init = thetas_phi.copy()

thetas_all_deg3 = np.rad2deg(thetas_all3[:, 0])

# convert to degrees
sim_deg3 = np.rad2deg(thetas_all3[:, 0])
# ============================
# 3. LAB DATA & COMPARISON
# ============================

lab_data_str = """\
-90  22.4794344
-89  21.71764379
-88  21.71764379
-87  21.55193742
-86  21.13423988
-85  21.13423988
-84  21.29735405
-83  21.13423988
-82  20.55604522
-81  20.81506597
-80  20.40020939
-79  20.40020939
-78  20.40020939
-77  19.83212985
-76  19.83212985
-75  19.83212985
-74  19.2692944
-73  19.2692944
-72  18.85315876
-71  18.71173788
-70  18.29621839
-69  18.29621839
-68  17.8786966
-67  17.74467163
-66  17.32792178
-65  17.74467163
-64  16.90927181
-63  17.32792178
-62  16.78264415
-61  16.36491817
-60  16.36491817
-59  15.82615408
-58  15.82615408
-57  15.40770364
-56  15.40770364
-55  14.875682
-54  14.875682
-53  14.4567554
-52  14.4567554
-51  14.03624347
-50  14.03624347
-49  13.51253064
-48  13.51253064
-47  13.09189306
-46  12.66981426
-45  12.66981426
-44  12.66981426
-43  12.24633286
-42  11.73308416
-41  11.30993247
-40  10.88552705
-39  10.88552705
-38  10.88552705
-37  10.45990909
-36  10.45990909
-35  9.958070598
-34  9.53323279
-33  9.53323279
-32  9.107334312
-31  8.680418425
-30  8.680418425
-29  8.252529047
-28  7.823710732
-27  7.394008643
-26  7.338606336
-25  7.338606336
-24  6.911227119
-23  6.483073693
-22  6.483073693
-21  5.624628041
-20  5.624628041
-19  5.194428908
-18  4.763641691
-17  4.763641691
-16  4.332313983
-15  3.900493742
-14  3.900493742
-13  3.871256232
-12  3.468229259
-11  3.012787504
-10  2.583020669
-9   2.602562202
-8   2.152962789
-7   1.735704589
-6   1.301952673
-5   0.861525735
-4   0.861525735
-3   0.430787217
-2   0
-1   0
0   -0.430787217
1   -0.861525735
2   -0.86805145
3   -1.301952673
4   -1.301952673
5   -1.735704589
6   -2.16925759
7   -2.602562202
8   -3.035569125
9   -3.035569125
10  -3.900493742
11  -3.900493742
12  -4.29986282
13  -4.727987819
14  -4.763641691
15  -4.727987819
16  -5.624628041
17  -5.667285678
18  -6.10005996
19  -6.483073693
20  -6.483073693
21  -6.911227119
22  -7.394008643
23  -7.394008643
24  -7.338606336
25  -8.190861376
26  -8.190861376
27  -8.680418425
28  -8.746162263
29  -9.107334312
30  -9.53323279
31  -9.605204155
32  -9.958070598
33  -10.38180516
34  -10.45990909
35  -10.88552705
36  -10.88552705
37  -10.88552705
38  -11.39532105
39  -11.73308416
40  -12.1549417
41  -12.1549417
42  -12.66981426
43  -12.66981426
44  -13.09189306
45  -13.51253064
46  -13.61418274
47  -13.61418274
48  -14.03624347
49  -14.03624347
50  -14.4567554
51  -14.98756197
52  -14.98756197
53  -15.40770364
54  -15.40770364
55  -15.9453959
56  -16.36491817
57  -16.36491817
58  -16.78264415
59  -16.90927181
60  -17.32792178
61  -17.32792178
62  -17.32792178
63  -17.8786966
64  -17.8786966
65  -18.29621839
66  -18.29621839
67  -18.85315876
68  -18.85315876
69  -19.2692944
70  -19.41546546
71  -19.83212985
72  -19.98310652
73  -19.98310652
74  -20.40020939
75  -20.71417484
76  -20.55604522
77  -20.97349342
78  -21.13423988
79  -21.13423988
80  -21.13423988
81  -22.13549188
82  -22.30620505
83  -22.30620505
84  -23.13946176
85  -22.89986653
86  -23.49856568
87  -23.49856568
88  -23.68208772
89  -23.68208772
90  -24.1022345
"""
lab_data_str2 = """\
-90	25.74070836
-89	25.74070836
-88	25.12782418
-87	24.9342951
-86	24.9342951
-85	24.9342951
-84	24.33124192
-83	24.33124192
-82	23.7329398
-81	23.7329398
-80	23.55226367
-79	23.13946176
-78	22.96377306
-77	22.96377306
-76	22.79059084
-75	22.38013505
-74	22.38013505
-73	21.80140949
-72	21.63794123
-71	21.22765119
-70	21.22765119
-69	20.65891006
-68	20.65891006
-67	20.24662008
-66	19.83212985
-65	19.68332755
-64	20.09523119
-63	19.53665494
-62	19.12522602
-61	19.12522602
-60	18.57234851
-59	18.15949047
-58	18.15949047
-57	17.74467163
-56	17.19854122
-55	17.19854122
-54	17.19854122
-53	16.78264415
-52	16.65784556
-51	16.2428791
-50	15.40770364
-49	15.70863783
-48	15.29298769
-47	14.875682
-46	14.875682
-45	14.4567554
-44	14.03624347
-43	13.51253064
-42	13.93168924
-41	13.51253064
-40	13.09189306
-39	12.66981426
-38	12.1549417
-37	12.1549417
-36	11.30993247
-35	11.30993247
-34	10.88552705
-33	10.45990909
-32	9.958070598
-31	9.605204155
-30	9.958070598
-29	9.53323279
-28	9.107334312
-27	8.615648184
-26	8.680418425
-25	8.252529047
-24	7.823710732
-23	7.394008643
-22	6.911227119
-21	6.483073693
-20	6.483073693
-19	6.054191894
-18	5.624628041
-17	5.194428908
-16	5.155584317
-15	4.763641691
-14	4.332313983
-13	3.871256232
-12	3.468229259
-11	3.442215286
-10	3.012787504
-9	2.583020669
-8	2.152962789
-7	1.722662072
-6	1.735704589
-5	1.301952673
-4	0.861525735
-3	0.430787217
-2	0
-1	-0.434050632
0	-0.861525735
1	-0.861525735
2	-1.735704589
3	-1.722662072
4	-2.16925759
5	-2.602562202
6	-2.583020669
7	-3.468229259
8	-3.442215286
9	-3.900493742
10	-4.29986282
11	-4.727987819
12	-5.155584317
13	-5.582605757
14	-5.624628041
15	-6.10005996
16	-6.483073693
17	-6.483073693
18	-6.911227119
19	-7.338606336
20	-7.765166018
21	-7.823710732
22	-8.190861376
23	-8.680418425
24	-9.107334312
25	-9.53323279
26	-9.53323279
27	-9.958070598
28	-10.38180516
29	-10.45990909
30	-11.30993247
31	-11.73308416
32	-11.73308416
33	-11.73308416
34	-12.1549417
35	-12.66981426
36	-13.09189306
37	-13.09189306
38	-13.51253064
39	-13.93168924
40	-14.03624347
41	-14.4567554
42	-14.875682
43	-14.875682
44	-15.40770364
45	-15.82615408
46	-16.2428791
47	-16.2428791
48	-16.78264415
49	-16.78264415
50	-17.19854122
51	-17.32792178
52	-17.74467163
53	-18.57234851
54	-18.15949047
55	-18.71173788
56	-19.12522602
57	-19.12522602
58	-19.68332755
59	-20.09523119
60	-20.09523119
61	-20.24662008
62	-20.65891006
63	-21.22765119
64	-21.22765119
65	-21.63794123
66	-22.21183069
67	-22.79059084
68	-22.96377306
69	-22.96377306
70	-23.37416437
71	-23.96248897
72	-23.96248897
73	-24.55549736
74	-24.55549736
75	-24.74353783
76	-25.15311714
77	-24.9342951
78	-25.9533757
79	-25.54202099
80	-26.56505118
81	-26.15433578
82	-26.77115021
83	-26.98023072
84	-26.98023072
85	-27.19235216
86	-27.60667785
87	-27.82409638
88	-28.04468695
89	-28.68614757
90	-28.68614757
"""
lab_data_str3 = """\
-90	29.14807185
-89	29.5665755
-88	29.10047741
-87	28.87242417
-86	28.87242417
-85	28.64761646
-84	28.01789355
-83	27.80145878
-82	27.60667785
-81	27.18111109
-80	27.18111109
-79	26.76750892
-78	26.36187551
-77	25.75527076
-76	26.16156644
-75	25.55996517
-74	25.36740446
-73	25.36740446
-72	24.77514057
-71	24.77514057
-70	24.18735352
-69	24.18735352
-68	23.60411504
-67	23.19859051
-66	23.02549201
-65	23.02549201
-64	22.45154646
-63	22.45154646
-62	21.88233567
-61	21.88233567
-60	21.47679098
-59	20.91254769
-58	20.91254769
-57	20.50497948
-56	20.35322915
-55	19.94599786
-54	19.94599786
-53	19.39206851
-52	18.98321706
-51	18.57234851
-50	18.02472352
-49	18.02472352
-48	17.61257784
-47	17.07102129
-46	16.65784556
-45	16.65784556
-44	16.2428791
-43	16.12259878
-42	15.70863783
-41	15.29298769
-40	14.875682
-39	14.4567554
-38	13.93168924
-37	13.93168924
-36	13.41235764
-35	13.51253064
-34	12.5754655
-33	12.1549417
-32	11.64597425
-31	11.22579776
-30	11.30993247
-29	10.45990909
-28	10.45990909
-27	10.03312055
-26	9.039482803
-25	9.462322208
-24	9.107334312
-23	8.680418425
-22	8.190861376
-21	7.765166018
-20	7.338606336
-19	6.963468526
-18	6.483073693
-17	6.009005957
-16	5.624628041
-15	5.194428908
-14	4.727987819
-13	4.29986282
-12	3.900493742
-11	3.442215286
-10	3.012787504
-9	2.583020669
-8	2.583020669
-7	2.152962789
-6	1.722662072
-5	1.292166886
-4	0.861525735
-3	0.430787217
-2	-0.434050632
-1	-0.861525735
0	-1.292166886
1	-1.722662072
2	-1.735704589
3	-2.152962789
4	-2.583020669
5	-3.012787504
6	-3.442215286
7	-3.930175546
8	-4.332313983
9	-4.727987819
10	-5.194428908
11	-5.194428908
12	-5.624628041
13	-6.532136688
14	-6.911227119
15	-7.338606336
16	-7.765166018
17	-8.252529047
18	-8.190861376
19	-8.680418425
20	-9.107334312
21	-9.53323279
22	-9.958070598
23	-10.38180516
24	-10.38180516
25	-10.88552705
26	-11.30993247
27	-11.73308416
28	-12.1549417
29	-12.5754655
30	-13.09189306
31	-13.09189306
32	-13.51253064
33	-14.34933204
34	-14.4567554
35	-14.875682
36	-15.29298769
37	-15.40770364
38	-15.82615408
39	-16.65784556
40	-16.65784556
41	-17.19854122
42	-17.61257784
43	-17.61257784
44	-18.15949047
45	-18.57234851
46	-18.98321706
47	-19.12522602
48	-19.53665494
49	-19.94599786
50	-20.50497948
51	-20.91254769
52	-21.47679098
53	-22.04591346
54	-22.45154646
55	-22.04591346
56	-22.61986495
57	-23.02549201
58	-23.60411504
59	-23.60411504
60	-24.18735352
61	-24.59011717
62	-24.77514057
63	-24.77514057
64	-25.36740446
65	-25.96406854
66	-25.96406854
67	-26.16156644
68	-26.56505118
69	-27.17026577
70	-27.37770277
71	-27.77962073
72	-27.99205304
73	-28.39301942
74	-28.61045967
75	-29.23281746
76	-29.45840479
77	-29.28142706
78	-29.68718164
79	-29.91920869
80	-30.15454792
81	-30.80144598
82	-31.04549083
83	-31.04549083
84	-31.70142967
85	-31.95436294
86	-32.21092772
87	-32.47119229
88	-32.73522627
89	-33.00310069
90	-32.85572195
"""
lab_data   = np.loadtxt(StringIO(lab_data_str))
lab_angles = lab_data[:, 0]   # -90 ... +90
lab_bend   = lab_data[:, 1]   # 181 values

phi_for_lab = lab_angles + 90.0  # align -90 → 0, +90 → 180

# interpolate simulation onto 181 lab points
sim_interp = np.interp(phi_for_lab, phis_deg, sim_deg)



lab_data2   = np.loadtxt(StringIO(lab_data_str2))
lab_angles2 = lab_data2[:, 0]   # -90 ... +90
lab_bend2   = lab_data2[:, 1]   # 181 values

phi_for_lab2 = lab_angles2 + 90.0  # align -90 → 0, +90 → 180

# interpolate simulation onto 181 lab points
sim_interp2 = np.interp(phi_for_lab2, phis_deg, sim_deg2)


lab_data2   = np.loadtxt(StringIO(lab_data_str2))
lab_angles2 = lab_data2[:, 0]   # -90 ... +90
lab_bend2   = lab_data2[:, 1]   # 181 values

phi_for_lab2 = lab_angles2 + 90.0  # align -90 → 0, +90 → 180

# interpolate simulation onto 181 lab points
sim_interp2 = np.interp(phi_for_lab2, phis_deg, sim_deg2)

lab_data3   = np.loadtxt(StringIO(lab_data_str3))
lab_angles3 = lab_data3[:, 0]   # -90 ... +90
lab_bend3   = lab_data3[:, 1]   # 181 values

phi_for_lab3 = lab_angles3 + 90.0  # align -90 → 0, +90 → 180

# interpolate simulation onto 181 lab points
sim_interp3 = np.interp(phi_for_lab3, phis_deg, sim_deg3)



rmse = np.sqrt(np.mean((sim_interp - lab_bend)**2))
print("RMSE between sim and lab (deg):", rmse)

# plot comparison
plt.figure()
plt.plot(phi_for_lab, lab_bend, 'k.', label='Lab data')
plt.plot(phi_for_lab2, lab_bend2, 'b.', label='Lab data2')
plt.plot(phi_for_lab3, lab_bend3, 'g.', label='Lab data3')

plt.plot(phi_for_lab, sim_interp, 'r-', label='Simulation')
plt.plot(phi_for_lab2, sim_interp2, 'c-', label='Simulation2')
plt.plot(phi_for_lab3, sim_interp3, 'm-', label='Simulation3')

plt.xlabel('Magnet rotation (shifted) [deg]')
plt.ylabel('Beam bending [deg]')
plt.title('Comparison of lab bending and simulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# 4. ENERGY LANDSCAPE (ELASTIC, GRAV, MAG)
# =========================================


def B_from_epm(p):
    """
    Magnetic field B at point p (3,) due to EPM dipole at p_epm.
    Dipole model: B = mu0/(4π r^3) (3 r_hat r_hat^T - I) μ_epm
    Uses mu_epm and mu0 from parameters.py
    """
    r_vec = p - r_epm
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-6:
        return np.zeros(3)
    r_hat = r_vec / r_norm
    A = 3.0 * np.outer(r_hat, r_hat) - np.eye(3)
    B = (mu0 / (4.0 * np.pi * r_norm**3)) * (A @ mu_epm)
    return B

def thetas_from_yz(y, z, lengths, L_cat):
    """
    Map a tip (y,z) to segment bending angles thetas
    assuming a single constant-curvature arc of total length L_cat.

    Total tip angle θ_tot is atan2(y, z),
    then each segment i gets θ_i = θ_tot * (L_i / L_cat) so that
    curvature κ = θ_i / L_i is constant.
    """
    z_safe = z if z > 1e-6 else 1e-6
    theta_tot = np.arctan2(y, z_safe)
    lengths = np.asarray(lengths, float)
    thetas = theta_tot * (lengths / float(L_cat))
    return thetas

def energies_for_tip_yz(
        y, z,
        axes=axes,
        lengths=lengths,
        masses=masses,
        gvec=gvec,
        E=E_arr,
        r=r_arr,
        nu=nu_arr,
        L_cat=L_cat
    ):
    """
    Compute:
      - Elastic bending energy U_el
      - Gravitational potential energy of catheter U_g_cat
      - Magnetic potential energy U_mag of tip IPM in EPM field

    for a given tip position (y,z).
    """
    # 1) map (y,z) to segment angles
    thetas = thetas_from_yz(y, z, lengths, L_cat)

    # 2) use energy_and_grad to get total U (elastic + grav)
    U_total, grad, K = energy_and_grad(
        axes, thetas, lengths, masses, gvec, E, r, nu,
        R_base=None, p_base=None, quad_n=12
    )

    # rebuild gamma to separate elastic part
    gamma = np.concatenate([
        (theta_i / L_i) * np.asarray(n_hat, float)
        for n_hat, theta_i, L_i in zip(axes, thetas, lengths)
    ])
    U_el = 0.5 * gamma @ (K @ gamma)
    U_g_cat = U_total - U_el

    # 3) magnetic energy of tip IPM in EPM field
    p_ipm = np.array([0.0, y, z])  # tip position in world
    B_here = B_from_epm(p_ipm)
    U_mag = - (mu_ipm @ B_here)

    return U_el, U_g_cat, U_mag

# Grid in y,z (in meters)
y_vals = np.linspace(-0.15, 0.15, 150)   # -150 to 150 mm
z_vals = np.linspace(0.02, 0.20, 150)    # 20 to 200 mm

Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
U_el_grid   = np.zeros_like(Y)
U_gcat_grid = np.zeros_like(Y)
U_mag_grid  = np.zeros_like(Y)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        U_el, U_gc, U_mag = energies_for_tip_yz(Y[i, j], Z[i, j])
        U_el_grid[i, j]   = U_el
        U_gcat_grid[i, j] = U_gc
        U_mag_grid[i, j]  = U_mag

# convert to mm for plotting
Ymm = Y * 1e3
Zmm = Z * 1e3

fig, ax_arr = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

# (a) Elastic energy
im0 = ax_arr[0].pcolormesh(Zmm, Ymm, U_el_grid, shading='auto', cmap='jet')
ax_arr[0].set_title('Elastic potential energy')
ax_arr[0].set_xlabel('z (mm)')
ax_arr[0].set_ylabel('y (mm)')
cbar0 = fig.colorbar(im0, ax=ax_arr[0])
cbar0.set_label('U_el (J)')

# (b) Catheter gravitational energy
im1 = ax_arr[1].pcolormesh(Zmm, Ymm, U_gcat_grid, shading='auto', cmap='jet')
ax_arr[1].set_title('Catheter gravitational energy')
ax_arr[1].set_xlabel('z (mm)')
cbar1 = fig.colorbar(im1, ax=ax_arr[1])
cbar1.set_label('U_g, catheter (J)')

# (c) Magnetic energy
im2 = ax_arr[2].pcolormesh(Zmm, Ymm, U_mag_grid, shading='auto', cmap='jet')
ax_arr[2].set_title('Magnetic potential energy')
ax_arr[2].set_xlabel('z (mm)')
cbar2 = fig.colorbar(im2, ax=ax_arr[2])
cbar2.set_label('U_mag (J)')

for ax in ax_arr:
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================
# 5. TOTAL ENERGY & EQUILIB
# ==========================

U_tot_grid = U_el_grid + U_gcat_grid + U_mag_grid

min_idx = np.unravel_index(np.argmin(U_tot_grid), U_tot_grid.shape)
y_eq_m  = Y[min_idx]
z_eq_m  = Z[min_idx]
y_eq_mm = Ymm[min_idx]
z_eq_mm = Zmm[min_idx]
U_min   = U_tot_grid[min_idx]

print("Equilibrium (approx) from total energy:")
print(f"  y_eq = {y_eq_mm:.2f} mm,  z_eq = {z_eq_mm:.2f} mm,  U_min = {U_min:.3e} J")

fig_tot, ax_tot = plt.subplots(figsize=(6, 5))
im_tot = ax_tot.pcolormesh(Zmm, Ymm, U_tot_grid, shading='auto', cmap='jet')
ax_tot.set_aspect('equal')
ax_tot.set_xlabel('z (mm)')
ax_tot.set_ylabel('y (mm)')
ax_tot.set_title('Total potential energy')
cbar_tot = fig_tot.colorbar(im_tot, ax=ax_tot)
cbar_tot.set_label('U_total (J)')

# mark equilibrium
ax_tot.plot(z_eq_mm, y_eq_mm, 'wo', markersize=8, markeredgecolor='k')
ax_tot.text(
    z_eq_mm + 5, y_eq_mm + 5,
    f"eq\n({z_eq_mm:.1f}, {y_eq_mm:.1f}) mm",
    color='w', fontsize=9, ha='left', va='bottom',
    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none')
)

ax_tot.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
