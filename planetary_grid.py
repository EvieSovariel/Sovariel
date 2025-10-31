import numpy as np

# Ley line grid (12 nodes: major sites)
ley_phases = np.array([0, 0.52, 1.05, 1.57, 2.09, 2.62, 3.14, 3.67, 4.19, 4.71, 5.24, 5.76])  # 3-6-9 scaled
N = 12
phases = np.random.normal(ley_phases, 0.01, N) % (2*np.pi)

# 3-6-9 coupling
weights = [3, 6, 9, 3, 6, 9, 3, 6, 9, 3, 6, 9]
K = 3.69

for step in range(3):
    mean = np.angle(np.mean(np.exp(1j * phases)))
    dtheta = K * np.sin(mean - phases)
    phases = (phases + dtheta) % (2 * np.pi)
    R = np.abs(np.mean(np.exp(1j * phases)))
    print(f"Grid Step {step+1}: R = {R:.6f}")

print("Planetary grid locked — breathe the resonance.")
