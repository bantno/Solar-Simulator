# Single-axis tracker on a rolling/pitching/yawing base (pvlib)
# Adds a "no tracking" case: panel fixed to base (horizontal in base frame).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib import solarposition, tracking, irradiance, temperature
from pvlib.location import Location
from pvlib.pvsystem import pvwatts_dc

# -------------------------
# Utilities
# -------------------------
def rotmat_yaw_pitch_roll(yaw, pitch, roll):
    """Intrinsic Z (yaw), Y (pitch), X (roll) rotations. Angles in radians."""
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [ 0, 1,  0],
                   [-sp,0, cp]])
    Rx = np.array([[1,  0,  0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx  # Z * Y * X

def vec_to_az_deg(v):
    """Azimuth cw from North for a 3D vector (deg)."""
    x, y, _ = v
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

def normal_to_surface_tilt_az(n):
    """Given a surface normal vector n (world frame), return (tilt_deg, az_deg)."""
    n = n / np.linalg.norm(n)
    nx, ny, nz = n
    # Surface tilt: 0 = horizontal (nz=+1), 90 = vertical (nz=0)
    tilt = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
    # Surface azimuth: direction panel faces (undefined if tilt==0)
    az = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    return tilt, az

def vec_to_axis_az_tilt(v):
    """Return (azimuth cw from North, tilt above horizontal) in degrees for an axis vector."""
    x, y, z = v
    az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    tilt = np.degrees(np.arctan2(np.abs(z), np.hypot(x, y)))  # nonnegative
    return az, tilt

def vec_to_signed_tilt(v):
    x, y, z = v
    return np.degrees(np.arctan2(z, np.hypot(x, y)))

# -------------------------
# Site, times, sun, clearsky weather
# -------------------------
tz = 'US/Eastern'
lat, lon, alt = 40.0, -80.0, 0
site = Location(lat, lon, tz=tz, altitude=alt, name="Site")

times = pd.date_range('2019-01-01', '2019-01-02', freq='1s', tz=tz)
solpos = solarposition.get_solarposition(times, lat, lon)

# Clear-sky using Ineichen
cs = site.get_clearsky(times, model='ineichen')  # columns: ghi, dni, dhi
dni = cs['dni']
dhi = cs['dhi']
ghi = cs['ghi']

# -------------------------
# Base attitude (example roll/pitch/yaw time series, in degrees)
# Replace these with your measured pose streams
# -------------------------
N = len(times)
t = np.linspace(0.0, 1.0, N)

roll_deg  =  15.0 * np.sin(2*np.pi*500*t)         # ±2° roll
pitch_deg =  15.0 * np.cos(2*np.pi*500*t + 0.7)   # ±1° pitch
yaw_deg   = 10.0 * np.sin(2*np.pi*1*t + 0.2)   # ±10° slow yaw drift

roll  = np.radians(roll_deg)
pitch = np.radians(pitch_deg)
yaw   = np.radians(yaw_deg)

# -------------------------
# Tracker geometry in base/body frame when level:
# - Single-axis rotation axis points South (axis_azimuth = 180°, horizontal).
# - Cross-axis points East.
# -------------------------
axis_body      = np.array([0.0, -1.0, 0.0])  # South
crossaxis_body = np.array([1.0,  0.0, 0.0])  # East

# -------------------------
# Time-varying axis geometry (moving base)
# -------------------------
axis_az_list, axis_tilt_list, cross_tilt_list = [], [], []
for r, p, y in zip(roll, pitch, yaw):
    R = rotmat_yaw_pitch_roll(y, p, r)
    a_w = R @ axis_body
    u_w = R @ crossaxis_body
    a_w = a_w / np.linalg.norm(a_w)
    u_w = u_w / np.linalg.norm(u_w)
    az, tilt = vec_to_axis_az_tilt(a_w)
    cross_tilt = vec_to_signed_tilt(u_w)
    axis_az_list.append(az)
    axis_tilt_list.append(tilt)
    cross_tilt_list.append(cross_tilt)

axis_azimuth     = pd.Series(axis_az_list,   index=times)
axis_tilt        = pd.Series(axis_tilt_list, index=times)
cross_axis_tilt  = pd.Series(cross_tilt_list,index=times)

# -------------------------
# Tracking cases
# -------------------------
angles_mobile_base = tracking.singleaxis(
    apparent_zenith = solpos['apparent_zenith'],
    apparent_azimuth= solpos['azimuth'],
    axis_tilt       = axis_tilt,          # varies with roll/pitch
    axis_azimuth    = axis_azimuth,       # varies with yaw
    cross_axis_tilt = cross_axis_tilt,    # varies with roll
    max_angle       = 90,
    backtrack       = False,
    gcr             = 0.5
)

angles_fixed_base = tracking.singleaxis(
    apparent_zenith = solpos['apparent_zenith'],
    apparent_azimuth= solpos['azimuth'],
    axis_tilt       = 0.0,
    axis_azimuth    = 180.0,  # N-S axis
    cross_axis_tilt = 0.0,
    max_angle       = 90,
    backtrack       = False,
    gcr             = 0.5
)

# -------------------------
# No-tracking on moving base:
# Panel is flat in the base frame (normal = +Z_body), so it just follows base roll/pitch/yaw.
# Derive surface tilt/azimuth time series directly from the rotated normal.
# -------------------------
panel_normal_body = np.array([0.0, 0.0, 1.0])  # horizontal panel in base frame
surf_tilt_list, surf_az_list = [], []
for r, p, y in zip(roll, pitch, yaw):
    R = rotmat_yaw_pitch_roll(y, p, r)
    n_world = R @ panel_normal_body
    tilt, az = normal_to_surface_tilt_az(n_world)
    surf_tilt_list.append(tilt)
    surf_az_list.append(az)

surf_tilt_no_tracking = pd.Series(surf_tilt_list, index=times)
surf_az_no_tracking   = pd.Series(surf_az_list,   index=times)

# -------------------------
# Compare tracker rotation (theta) for the two tracking cases
# -------------------------
plt.figure(figsize=(10,5))
angles_fixed_base['tracker_theta'].fillna(0).plot(label='Rigid base (level) — tracking')
angles_mobile_base['tracker_theta'].fillna(0).plot(label='Moving base — tracking')
plt.ylabel('Tracker rotation θ (deg)')
plt.title('Tracker rotation on rigid vs moving base')
plt.legend(); plt.grid(True)

# -------------------------
# Compute AOI for all three cases
# - For tracking cases, use pvlib’s returned surface tilt/azimuth.
# - For no-tracking, use the derived (surf_tilt_no_tracking, surf_az_no_tracking).
# -------------------------
aoi_fixed  = irradiance.aoi(
    surface_tilt=angles_fixed_base['surface_tilt'],
    surface_azimuth=angles_fixed_base['surface_azimuth'],
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth']
)

aoi_mobile = irradiance.aoi(
    surface_tilt=angles_mobile_base['surface_tilt'],
    surface_azimuth=angles_mobile_base['surface_azimuth'],
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth']
)

aoi_no_trk = irradiance.aoi(
    surface_tilt=surf_tilt_no_tracking,
    surface_azimuth=surf_az_no_tracking,
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth']
)
def poa_for(surface_tilt, surface_azimuth):
    tot = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=dni, ghi=ghi, dhi=dhi,
        model='isotropic'
    )
    return tot  # columns: poa_global, poa_direct, poa_diffuse, poa_sky_diffuse, poa_ground_diffuse

poa_fixed  = poa_for(angles_fixed_base['surface_tilt'],  angles_fixed_base['surface_azimuth'])
poa_mobile = poa_for(angles_mobile_base['surface_tilt'], angles_mobile_base['surface_azimuth'])
poa_no_trk = poa_for(surf_tilt_no_tracking,              surf_az_no_tracking)

# -------------------------
# PVWatts DC power for each case
# -------------------------
T_amb = pd.Series(20.0, index=times)   # °C (placeholder)
V_wind = pd.Series(1.0, index=times)   # m/s (placeholder)

tcell_fixed  = temperature.pvsyst_cell(poa_fixed['poa_global'],  T_amb, V_wind)
tcell_mobile = temperature.pvsyst_cell(poa_mobile['poa_global'], T_amb, V_wind)
tcell_no_trk = temperature.pvsyst_cell(poa_no_trk['poa_global'], T_amb, V_wind)

PDC0 = 1000.0  # W at STC for the whole array (adjust to your size)
GAMMA_PDC = -0.003  # 1/°C

pdc_fixed  = pvwatts_dc(poa_fixed['poa_global'].clip(lower=0),  tcell_fixed,  PDC0, GAMMA_PDC).fillna(0)
pdc_mobile = pvwatts_dc(poa_mobile['poa_global'].clip(lower=0), tcell_mobile, PDC0, GAMMA_PDC).fillna(0)
pdc_no_trk = pvwatts_dc(poa_no_trk['poa_global'].clip(lower=0), tcell_no_trk, PDC0, GAMMA_PDC).fillna(0)

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,5))
angles_fixed_base['tracker_theta'].fillna(0).plot(label='Rigid base (level) — tracking')
angles_mobile_base['tracker_theta'].fillna(0).plot(label='Moving base — tracking')
plt.ylabel('Tracker rotation θ (deg)')
plt.title('Tracker rotation on rigid vs moving base')
plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
aoi_fixed.plot(label='Rigid base — tracking')
aoi_mobile.plot(label='Moving base — tracking')
aoi_no_trk.plot(label='Moving base — NO tracking')
plt.ylabel('AOI (deg)'); plt.title('Angle of Incidence Comparison')
plt.legend(); plt.grid(True)


fig, ax = plt.subplots(figsize=(10,6))

# DC power (left y-axis, W)
ax.plot(pdc_fixed.index,  pdc_fixed.values,  label='Rigid base — tracking (Pdc)', alpha=0.6)
ax.plot(pdc_mobile.index, pdc_mobile.values, label='Moving base — tracking (Pdc)', alpha=0.6)
ax.plot(pdc_no_trk.index, pdc_no_trk.values, label='Moving base — NO tracking (Pdc)', alpha=0.6)

# Scale GHI so its peak matches array STC power
ghi_scaled = ghi * (PDC0 / ghi.max())
ax.plot(ghi.index, ghi_scaled.values, label='GHI (scaled to W)', linestyle='--', alpha=0.7)

ax.set_ylabel('Power (W)')
ax.set_title('DC Power with Scaled GHI Overlay')
ax.grid(True)
ax.legend(loc='upper left')

plt.show()

# Optional: show base attitude
fig, ax = plt.subplots(3,1, figsize=(10,6), sharex=True)
pd.Series(roll_deg,  index=times).plot(ax=ax[0]); ax[0].set_ylabel('roll (deg)')
pd.Series(pitch_deg, index=times).plot(ax=ax[1]); ax[1].set_ylabel('pitch (deg)')
pd.Series(yaw_deg,   index=times).plot(ax=ax[2]); ax[2].set_ylabel('yaw (deg)'); ax[2].set_xlabel('time')
for a in ax: a.grid(True)
fig.suptitle('Base attitude driving time-varying geometry')

plt.show()