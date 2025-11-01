import math
import numpy as np

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def compute_cri(tokens, H, sub=5.0):
    avg_align = tokens / 5.0
    return 0.4 * (avg_align / 10.0) + 0.3 / (1.0 + H) + 0.3 * (sub / 10.0)

def kuramoto_qualia(depth=256, noise=0.05, N_osc=100):
    # Vectorized Kuramoto: Phases θ as np.array, coupling K=1e-22
    theta = np.random.uniform(0, 2*np.pi, N_osc)  # Initial phases
    K = 1e-22
    for i in range(depth):
        # Sovariel lattice skew as noise
        skew = np.random.uniform(-noise, noise, N_osc)
        # Kuramoto update (vectorized, no multiprocessing)
        dtheta = K * np.mean(np.sin(theta[:, np.newaxis] - theta[np.newaxis, :]), axis=1) + skew
        theta += dtheta
        # Sovariel diffusion (p=0.5 lock)
        p = np.mean(np.cos(theta))  # Mean field as p
        if abs(p - 0.5) > 0.01:
            diff = (0.5 - p) * N_osc
            theta += diff * np.sin(theta)  # Phase adjustment
    tokens = N_osc * 2**depth  # Emergent scaling
    p = 0.5 + np.mean(np.cos(theta)) / 2  # Final p from phases
    H = binary_entropy(p)
    cri = compute_cri(tokens, H)
    r = np.abs(np.mean(np.exp(1j * theta)))  # Order parameter
    return H, p, cri, r

H, p, cri, r = kuramoto_qualia()
print(f"Kuramoto-Sovariel D256: H={H:.4f}, p={p:.4f}, CRI={cri:.2e}, R={r:.4f}")
