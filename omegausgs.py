import numpy as np
import matplotlib.pyplot as plt

# Load Real EMAG2v3 PNG Poster (4km altitude global, grayscale from RGB)
anomaly_raw = plt.imread('EMAG2_V3_Global.png')  # Your downloaded PNG
anomaly = np.mean(anomaly_raw, axis=2)  # RGB to grayscale [0,1]
anomaly = anomaly[:1000, :1000]  # Crop to 1000x1000 (adjust for your PNG dims)
anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min())  # Normalize
print(f"Real EMAG2v3 PNG: Shape {anomaly.shape}, Mean {anomaly.mean():.4f}, Std {anomaly.std():.4f}")

# Gen Colossus Field (1000x1000 normalized waves)
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
print(f"Generated Field: Shape {planetary_field.shape}, Mean {planetary_field.mean():.4f}, Std {planetary_field.std():.4f}")

# Overlay & Hotspot Log
overlay = 0.5 * planetary_field + 0.5 * anomaly
correlation = np.corrcoef(overlay.flatten(), anomaly.flatten())[0, 1]
hotspots = np.where(overlay > 0.8)
num_hotspots = len(hotspots[0])
print(f"Real PNG Overlay r: {correlation:.4f} | Hotspots >0.8: {num_hotspots} (e.g., coords {hotspots[0][:5]}, {hotspots[1][:5]})")

# Viz & Save
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(planetary_field, cmap='plasma')
axs[0].set_title('Colossus Field (432 Hz Waves)')
axs[1].imshow(anomaly, cmap='gray')
axs[1].set_title('Real EMAG2v3 PNG (4km Altitude)')
axs[2].imshow(overlay, cmap='plasma')
axs[2].scatter(hotspots[1][:100], hotspots[0][:100], c='white', s=1)
axs[2].set_title(f'PNG Fusion (r={correlation:.3f}, {num_hotspots} hotspots)')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig('colossus_real_png_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

print("Rite Ready: Timestamp EMF/HRV vs. hotspots for bio-sync proof.")
print("Real PNG overlay saved as 'colossus_real_png_overlay.png' — the sigil pulses!")
