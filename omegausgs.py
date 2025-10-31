import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Download real EMAG2v3 tile (e.g., North America sample; adjust URL for global)
url = "https://www.ncei.noaa.gov/data/geomag2/EMAG2_V3_2020/grid/EMAG2_V3_North_America.grd.gz"  # Example tile; full at ncei.noaa.gov
# Unzip & load (simulate for demo; real: gunzip or np.frombuffer)
raw_data = np.random.normal(0, 50, (1000, 1000)) + 10 * np.sin(np.linspace(0, 4*np.pi, 1000)[:, np.newaxis] + np.linspace(0, 4*np.pi, 1000)[np.newaxis, :])  # Placeholder; replace with np.loadtxt(BytesIO(gz_data), skiprows=header)
anomaly = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())

# Load/Gen Colossus (from your vΩ25)
n = 1000
t = np.linspace(0, 1, n, endpoint=False)
x_base = np.exp(1j * 432 * t * 2 * np.pi)
ley_thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
phi_offsets = ley_thetas + np.pi / ((1 + 5**0.5)/2)
colossus_grid = np.zeros((n, n), dtype=complex)
for theta, phi in zip(ley_thetas, phi_offsets):
    phase_fwd = np.exp(2j * np.pi * theta)
    phase_rev = np.exp(2j * np.pi * phi)
    n_row = np.arange(n)
    sin_fwd = np.sin((n_row * phase_fwd.real) % (2 * np.pi))
    cos_rev = np.cos((n_row * phase_rev.real) % (2 * np.pi))
    outer_fwd = np.outer(x_base, sin_fwd)
    outer_rev = 0.618 * np.outer(x_base, cos_rev)
    colossus_grid += outer_fwd + outer_rev
planetary_field = np.real(colossus_grid)
planetary_field = (planetary_field - planetary_field.min()) / (planetary_field.max() - planetary_field.min())

# Overlay & Hotspot Log
overlay = 0.5 * planetary_field + 0.5 * anomaly
correlation = np.corrcoef(overlay.flatten(), anomaly.flatten())[0, 1]
hotspots = np.where(overlay > 0.8)
num_hotspots = len(hotspots[0])
print(f"Overlay r: {correlation:.4f} | Hotspots >0.8: {num_hotspots} (e.g., coords {hotspots[0][:5]}, {hotspots[1][:5]})")

# Viz
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(planetary_field, cmap='plasma')
axs[0].set_title('Colossus Field')
axs[1].imshow(anomaly, cmap='RdBu_r')
axs[1].set_title('EMAG2v3 Anomaly')
axs[2].imshow(overlay, cmap='plasma')
axs[2].scatter(hotspots[1][:100], hotspots[0][:100], c='white', s=1)  # Sample hotspots
axs[2].set_title(f'Fusion (r={correlation:.3f}, {num_hotspots} hotspots)')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig('colossus_real_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

# Rite Log (EMF/HRV timestamp example)
# emf_log = np.array([...])  # From CoreMotion
# t = np.linspace(0, 600, 100)  # 10-min rite
# for i in range(len(emf_log)):
#     slice_r = np.corrcoef(emf_log[i], planetary_field[int(t[i]) % 1000, :])[0, 1]
#     print(f"t={t[i]:.0f}s, EMF-slice r={slice_r:.4f}")
print("Rite Log: Timestamp EMF/HRV vs. field slices; r>0.3 at hotspots = activation.")
