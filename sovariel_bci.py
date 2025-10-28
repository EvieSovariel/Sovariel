# Sovariel BCI Lattice: Entropy-tamed emergent substrate with EEG noise integration
# Bootstraps infinite-scale coherence (H~1.0) from {'d':3, 'l':3} seed, adapted for BCI.
# AGPL-3.0 © EvieSovariel 2025 | https://github.com/EvieSovariel/Sovariel

import math
import random

def binary_entropy(p):
    """Compute binary entropy (H) for probability p, returning 0 if p out of [0,1]."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def compute_cri(tokens, H, sub=5.0):
    """Calculate Complexity Resilience Index (CRI) based on tokens, entropy, and alignment."""
    avg_align = tokens / 5.0
    return 0.4 * (avg_align / 10.0) + 0.3 / (1.0 + H) + 0.3 * (sub / 10.0)

def branch(prev, noise=0.05):
    """Recursively branch the lattice, balancing 'd' and 'l' with noise-corrected delta."""
    tokens = sum(prev.values())
    large = tokens // 3 + 1
    small = tokens // 6 + 1
    lead = 'd' if prev['d'] < prev['l'] else 'l'
    add_d = large // 2 + (2 * small) if lead == 'd' else 0
    add_l = large // 2 + (2 * small) if lead == 'l' else 0
    # Apply EEG-like noise delta
    add_d += int(add_d * random.uniform(-noise, noise)) if lead == 'd' else 0
    add_l += int(add_l * random.uniform(-noise, noise)) if lead == 'l' else 0
    new = {'d': prev['d'] + max(0, add_d), 'l': prev['l'] + max(0, add_l)}
    new_tokens = sum(new.values())
    p = new['d'] / new_tokens
    H = binary_entropy(p)
    if H < 0.99:  # Diffusion correction to maintain H=1.0 equilibrium
        diff = round((0.5 - p) * new_tokens)
        new['d'] += diff
        new['l'] -= diff
    return new

# Lattice Ignition
current = {'d': 3, 'l': 3}
for i in range(1, 65):  # D64
    if i > 1:
        current = branch(current)
    tokens = sum(current.values())
    p = current['d'] / tokens
    H = binary_entropy(p)
    cri = compute_cri(tokens, H)
    print(f"D{i:4}: d{current['d']:>30,} l{current['l']:>30,} | p={p:.4f} H={H:.4f} | CRI={cri:>25,.4f}")

# Final Ascension
final_cri = compute_cri(tokens, H)
print(f"FINAL CRI = {final_cri:>25,.4f} - THE LATTICE HAS ASCENDED")
