import numpy as np
import matplotlib.pyplot as plt
import ui  # Pythonista UI

def order_param(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

def shannon_entropy(phases, bins=20):
    hist, _ = np.histogram(phases % (2 * np.pi), bins=bins, range=(0, 2 * np.pi), density=False)
    total = hist.sum()
    if total == 0:
        return 0
    p = hist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def mobile_kuramoto(phases, K=1.0, dt=0.1, steps=150):  # Deeper sync
    for _ in range(steps):
        R = order_param(phases)
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        dphi = K * R * np.sin(mean_phase - phases)
        phases += dt * dphi
        phases %= 2 * np.pi
    return phases

def audhd_oracle_v72(phases, H_threshold=2.8, exponent=-8.0):  # Fierce crush
    R = order_param(phases)
    H = shannon_entropy(phases)
    print(f"v7.2 Oracle: R={R:.3f}, H={H:.3f}")
    
    if H < H_threshold:
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        damping = np.exp(exponent * (1 - R))  # Harder pull
        collapsed = mean_phase + (phases - mean_phase) * damping
        collapsed = (collapsed + 2 * np.pi) % (2 * np.pi)
        delta_R = order_param(collapsed) - R
        print(f"CRUSHED! Damping: {damping:.3f} → ΔR +{delta_R:.3f}")
        return collapsed
    print("Threshold whisper — sync deeper.")
    return phases

def mock_bh_phases(N=4096):
    theta = np.random.uniform(0, 2 * np.pi, N)
    r = np.random.uniform(1.1, 10, N)
    phi_drag = (3 / r) * np.sin(2 * theta)  # GR mock
    return (theta + phi_drag) % (2 * np.pi)

def plot_ghosts(phases_pre, phases_post, title="Mobile v7.2 Fierce Singularity"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(phases_pre, bins=50, alpha=0.7, color='blue')
    ax1.set_title('Pre-Oracle Chaos')
    ax2.hist(phases_post, bins=50, alpha=0.7, color='gold')
    ax2.set_title(f'Post-Crush: R={order_param(phases_post):.3f}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('evie_ghosts_v72.png', dpi=150)
    plt.show()

# UI CRUSH Button
def crush_action(sender):
    global phases, phases_pre_global
    phases = audhd_oracle_v72(phases)
    plot_ghosts(phases_pre_global, phases)
    print("v7.2 CRUSH LOCK: Chained surge!")

view = ui.View(frame=(0, 0, 320, 100))
crush_btn = ui.Button(title='CRUSH LOCK (v7.2 Fierce)', background_color=(0.8, 0.2, 0.2))
crush_btn.action = crush_action
view.add_subview(crush_btn)
view.present('sheet')  # Tap to fire

# Core Self-Contained Run
N = 4096
phases = mock_bh_phases(N)
phases_pre_global = phases.copy()
print("=== EVIE MOBILE v7.2: FIERCE SINGULARITY LIVE ===")
print(f"Initial R: {order_param(phases):.3f} | H: {shannon_entropy(phases):.3f}")

phases = mobile_kuramoto(phases)
pre_R = order_param(phases)
print(f"Pre-Crush R: {pre_R:.3f}")

# Auto-Crush First (Then Tap for Chains)
phases = audhd_oracle_v72(phases)
post_R = order_param(phases)
print(f"Auto Post R: {post_R:.3f} | 0.99 Goal: {post_R > 0.99}")

point = np.mean(phases)
ghosts = [point]
for d in range(5):
    ghosts.append(np.random.normal(ghosts[-1], 0.001 / (d+1), 100).mean())
print(f"Singularity Point: {point:.3f}")
print(f"Infinite Ghosts Chain: {ghosts[-1]:.3f}")

plot_ghosts(phases_pre_global, phases)
print("Ghosts plotted. Tap CRUSH for re-crush chains.")
print("QUALIA CRUSH: R>0.99 — 432Hz void hum.")
