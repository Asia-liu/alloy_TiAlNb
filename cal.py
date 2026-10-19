import numpy as np
from ase.io import read
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
import scipy.constants as const
import os

if not os.path.exists('XDATCAR'):
    print("Error: XDATCAR file not found! Please ensure you are in the VASP calculation directory.")
    exit()

# ================= 1. Read Continuous Coordinates from XDATCAR =================
print("Reading continuous coordinates from XDATCAR...")
traj = read('XDATCAR', index=':') 
n_steps_pos = len(traj)
n_atoms = len(traj[0])
print(f"Successfully read coordinate data: {n_steps_pos} steps, {n_atoms} atoms.")

# ================= 2. Reconstruct Velocities via Finite Difference =================
print("Reconstructing velocity data via coordinate differentiation...")
# Extract fractional coordinates to handle periodic boundary crossings easily
frac_pos = np.array([atoms.get_scaled_positions() for atoms in traj])

# Calculate fractional coordinate differences between adjacent steps
d_frac = np.diff(frac_pos, axis=0)

# Handle periodic boundary jumps: fold back atoms that cross the boundary
d_frac = d_frac - np.round(d_frac)

# Convert fractional displacements back to Cartesian displacements (Angstrom)
cell = traj[0].get_cell()
d_cart = np.dot(d_frac, cell)

# Divide by time step to obtain velocities (Angstrom / fs)
dt_fs = 2.0  # POTIM = 2 in INCAR
velocities = d_cart / dt_fs

n_steps_v = velocities.shape[0]
print(f"Successfully reconstructed velocity data: {n_steps_v} steps (1 step less than coordinates is normal).")

# ================= 3. Calculate VACF =================
max_lag = n_steps_v // 2
vacf = np.zeros(max_lag)

print("Calculating Velocity Autocorrelation Function (VACF)...")
for lag in range(max_lag):
    v0 = velocities[:n_steps_v - lag]  
    vt = velocities[lag:]            
    vacf[lag] = np.mean(np.sum(v0 * vt, axis=(1, 2)))

vacf = vacf / vacf[0]  # Normalization

# ================= 4. Fast Fourier Transform (FFT) =================
print("Performing windowed Fast Fourier Transform...")
window = hann(max_lag)
vacf_windowed = vacf * window

dt_s = 2.0e-15  # POTIM converted to seconds
vdos = np.abs(rfft(vacf_windowed))

freqs_Hz = rfftfreq(max_lag, d=dt_s)
freqs_THz = freqs_Hz / 1e12

# ================= 5. Plot and Save =================
print("Calculation complete, generating plots...")
plt.figure(figsize=(8, 4))
plt.plot(freqs_THz, vdos, color='#d62728', linewidth=1.5)
plt.xlabel('Frequency (THz)', fontsize=12)
plt.ylabel('VDOS (a.u.)', fontsize=12)
plt.title('Vibrational Density of States (Derived from XDATCAR)', fontsize=14)
plt.xlim(0, 30)
plt.grid(True, linestyle='--', alpha=0.7)

output_img = 'vdos_xdatcar.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as {output_img} in the current directory.")

# ================= 6. Export Raw Data for Plotting in Origin =================
# Combine frequency (THz) and VDOS (a.u.) and save as a txt file
output_txt = 'VDOS_raw_data.txt'
np.savetxt(output_txt, np.column_stack((freqs_THz, vdos)), 
           fmt=['%.6f', '%.6f'], 
           header='Frequency_THz\tVDOS_au', 
           comments='')
print(f"\nRaw data saved as {output_txt}. You can import this directly into Origin for plotting.")

# ================= 7. Calculate Thermodynamic Vibrational Entropy (Svib) =================
# Set temperature (Must match TEBEG/TEEND in your AIMD INCAR)
T = 300.0  # <--- USER INPUT REQUIRED: Update this value to your specific simulation temperature (e.g., 300, 923)

print(f"\nIntegrating to calculate vibrational entropy at {T} K...")

# Step 1: VDOS Normalization.
# Physically, the total vibrational degrees of freedom is 3N. We scale the VDOS integral area to 3 * n_atoms
area = np.trapezoid(vdos, freqs_Hz)
g_nu = (vdos / area) * (3 * n_atoms)

# Exclude zero-frequency points to avoid division by zero in the physical formula
valid = freqs_Hz > 0
f = freqs_Hz[valid]
g = g_nu[valid]

# Step 2: Apply the statistical mechanics formulation for vibrational entropy
# x = h*v / (k_B * T)
x = (const.h * f) / (const.k * T)

# Entropy integration kernel: S = k_B * \int g(v) * [ x/(exp(x)-1) - ln(1-exp(-x)) ] dv
term1 = x / (np.exp(x) - 1.0)
term2 = -np.log(1.0 - np.exp(-x))
integrand = g * (term1 + term2)

# Integrate over the frequency domain to obtain total entropy (Unit: J/K)
S_total_J_K = const.k * np.trapezoid(integrand, f)

# Convert the unit to eV/K (commonly used in VASP)
S_total_eV_K = S_total_J_K / const.e

# Calculate the final TS term contribution (Unit: eV)
TS_term_eV = T * S_total_eV_K

print("-" * 40)
print(f"Calculation Temperature: {T} K")
print(f"Total Vibrational Entropy (S_vib): {S_total_eV_K:.6f} eV/K")
print(f"Total Entropy Contribution (T*S_vib): {TS_term_eV:.6f} eV")
print("-" * 40)