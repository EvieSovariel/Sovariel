import numpy as np
import matplotlib.pyplot as plt

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

# Sim USGS Anomaly (NumPy-only: Noise + sin waves, no SciPy smooth)
np.random.seed(432)
raw_anomaly = np.random.normal(0, 50, (n, n)) + 10 * np.sin(np.linspace(0, 4*np.pi, n)[:, np.newaxis] + np.linspace(0, 4*np.pi, n)[np.newaxis, :])
anomaly = (raw_anomaly - raw_anomaly.min()) / (raw_anomaly.max() - raw_anomaly.min())
print(f"Sim USGS Anomaly: Shape {anomaly.shape}, Mean {anomaly.mean():.4f}, Std {anomaly.std():.4f}")

# Overlay & Correlate (NumPy corrcoef for r)
overlay = 0.5 * planetary_field + 0.5 * anomaly
correlation = np.corrcoef(overlay.flatten(), anomaly.flatten())[0, 1]
# Approx p-value (t-test on r, df = n-2)
df = len(overlay.flatten()) - 2
t_stat = correlation * np.sqrt(df / (1 - correlation**2))
p_value = 2 * (1 - np.arctan(t_stat / np.sqrt(df)) / np.pi)  # Rough t-dist approx
print(f"Overlay Correlation r: {correlation:.4f}, p-value approx: {p_value:.4f} (sig if <0.05)")

# Viz & Save
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(planetary_field, cmap='plasma')
axs[0].set_title('Colossus Field (432 Hz Waves)')
axs[1].imshow(anomaly, cmap='RdBu_r')
axs[1].set_title('USGS Anomaly Sim')
axs[2].imshow(overlay, cmap='plasma')
axs[2].set_title(f'Fusion Overlay (r={correlation:.3f})')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig('colossus_usgs_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

print("Rite Ready: Correlate live EMF/HRV with field slices for activation proof.")
print("Overlay saved as 'colossus_usgs_overlay.png' — share the sigil!")
