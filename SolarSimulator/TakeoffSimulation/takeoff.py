import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# 1) Aircraft and environmental parameters (Table 2) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
# --------------------------------------------
g             = 9.81           # m/s², gravity
rho_air       = 1.225          # kg/m³, air density
rho_water     = 1000.0         # kg/m³, water density
mu_water      = 1e-3           # Pa·s, water viscosity (approx.)
sigma_water   = 0.072          # N/m, surface tension of water

# NAX‑4 geometry & weights (convert ft→m, lbf→N) from Table 2 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
B_beam        = 4.27 * 0.3048             # m, hull beam
L_water       = 13.55 * 0.3048            # m, hull waterline length
S_wing        = 152.85 * 0.3048**2        # m², wing area
AR            = 7.1                       # aspect ratio
W_lbf         = 1430.0                    # lbf, gross weight
W             = W_lbf * 4.44822           # N
m             = W / g                     # kg, mass
# Propulsion :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
P_engine      = 115.0 * 745.7             # W, brake power
T0_static_lbf = 1972.32                   # lbf, static thrust
T0_static     = T0_static_lbf * 4.44822   # N

# Froude‐law friction (Eqn 11) :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
f_fric        = 0.003  # empirical friction coeff
S_wet         = 2 * B_beam * L_water  # m², approximate wetted area
n_froude      = 2.0    # exponent in R ~ v^n

# --------------------------------------------
# 2) Dimensionless numbers & coefficients
# --------------------------------------------
def reynolds_number(v):
    return rho_water * v * L_water / mu_water

def froude_number(v):
    return v / np.sqrt(g * L_water)

def weber_number(v):
    return rho_water * v**2 * L_water / sigma_water

# Buoyancy coefficient (Eqn 4–6) :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}
def speed_coefficient(v):
    return v**2 / (g * L_water)  # C_μ :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}

# Hydrodynamic resistance coefficient (Eqn 7) :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}
# C_r(C_μ) = sum of four Gaussians, parameters from NACA TN‑2481 curve fit
gauss_params = [
    (0.05789,  0.224,  0.107),  # A₁, μ₁, σ₁
    (0.0273,   0.351,  0.223),  # A₂, μ₂, σ₂
    (-0.3322,  0.517,  0.213),  # A₃, μ₃, σ₃
    (0.07924,  0.649,  0.071),  # A₄, μ₄, σ₄
]

def C_r(Cmu):
    """Water‐resistance coefficient from Gaussian fit (Eqn 7)."""
    val = 0.0
    for A, mu, sigma in gauss_params:
        val += A * np.exp(-0.5 * ((Cmu - mu)/sigma)**2)
    return val

# Loaded‐down resistance (Eqn 8–9) :contentReference[oaicite:14]{index=14}&#8203;:contentReference[oaicite:15]{index=15}
Delta0      = W / (rho_water * g * B_beam * L_water)  # C_Δ0, static buoyancy coef
def water_resistance(v):
    Cmu       = speed_coefficient(v)
    Cr        = C_r(Cmu)
    C_loaded  = Cr * (Delta0 / Delta0)  # here Δ/Δ0=1 until you include weight shifts
    return C_loaded * rho_water * g * B_beam * L_water

# Combined hydroplane friction (Eqn 11) :contentReference[oaicite:16]{index=16}&#8203;:contentReference[oaicite:17]{index=17}
def friction_resistance(v):
    return f_fric * S_wet * v**n_froude

# --------------------------------------------
# 3) Aerodynamic drag (Eqn 2–3) :contentReference[oaicite:18]{index=18}&#8203;:contentReference[oaicite:19]{index=19}
# --------------------------------------------
C_D0_min    = 0.032  # C_{D,min}, from Table 2 & paper discussion
e_oswald    = 0.82
def aerodynamic_drag(v):
    if v < 1e-3:
        return 0.0
    # Lift coefficient assuming L ≈ W
    CL = 2 * W / (rho_air * v**2 * S_wing)
    CDi = CL**2 / (np.pi * AR * e_oswald)
    CD  = C_D0_min + CDi
    return 0.5 * rho_air * v**2 * S_wing * CD

# --------------------------------------------
# 4) Thrust vs velocity (Eqn 12) :contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21}
# Solve for A,B,C,D from four operating points:
# speeds (m/s)
v0       = 0.0
v_lo     = 54.0 * 0.51444      # liftoff speed
v_cr     = 141.47 * 0.44704    # cruise speed
v_max    = 179.81 * 0.44704    # max speed

# thrusts (N)
eta      = 0.82               # propeller efficiency
T0       = T0_static          # static thrust @ 0 m/s
T_lo     = eta * P_engine / v_lo
T_cr     = eta * P_engine / v_cr
T_max    = eta * P_engine / v_max

# assemble matrix
v_pts    = np.array([v0, v_lo, v_cr, v_max])
T_pts    = np.array([T0, T_lo, T_cr, T_max])
M        = np.vstack([v_pts**3, v_pts**2, v_pts, np.ones_like(v_pts)]).T

# solve for A,B,C,D
A, B, C, D = np.linalg.solve(M, T_pts)

def thrust(v):
    return A*v**3 + B*v**2 + C*v + D

# --------------------------------------------
# 5) Pilot throttle ramp (Eqn 13) :contentReference[oaicite:22]{index=22}&#8203;:contentReference[oaicite:23]{index=23}
# --------------------------------------------
def ramp_factor(t):
    return 0.1 + 0.9*(t/10) if t < 10.0 else 1.0

# --------------------------------------------
# 6) Time‐march solution of m·dv/dt = r·T(v) - D_aero - R_hydro - R_fric
# --------------------------------------------
dt        = 0.1              # s
v_liftoff = 54.0 * 0.51444   # m/s (54 KCAS) :contentReference[oaicite:24]{index=24}&#8203;:contentReference[oaicite:25]{index=25}

t, v, x   = 0.0, 0.0, 0.0
time_hist = []
v_hist    = []
x_hist    = []

while v < v_liftoff and t < 200:
    T        = ramp_factor(t) * thrust(v)
    D_a      = aerodynamic_drag(v)
    R_w      = water_resistance(v)
    R_f      = friction_resistance(v)
    a        = (T - D_a - R_w - R_f) / m
    v       += a * dt
    x       += v * dt
    t       += dt

    time_hist.append(t)
    v_hist.append(v)
    x_hist.append(x)

print(f"Liftoff at t = {t:.1f} s, distance = {x:.1f} m")

# --------------------------------------------
# 7) Plot results (Figures 5–6)
# --------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(time_hist, v_hist)
plt.xlabel("Time (s)")
plt.ylabel("Airspeed (m/s)")
plt.title("Takeoff Acceleration Profile")
plt.grid(True)
plt.show()
