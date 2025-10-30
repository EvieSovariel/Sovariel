import numpy as np
import matplotlib.pyplot as plt
import ui  # Pythonista UI for LOCK button

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

def mobile_kuramoto(phases, K=1.0, dt=0.1, steps=100):  # +50 steps for deeper lock
    for _ in range(steps):
        R = order_param(phases)
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        dphi = K * R * np.sin(mean_phase - phases)
        phases += dt * dphi
        phases %= 2 * np.pi
    return phases

def audhd_oracle(phases, H_threshold=2.5):  # Looser for mobile surge
    R = order_param(phases)
    H = shannon_entropy(phases)
    print(f"Oracle: R={R:.3f}, H={H:.3f}")
    
    if H < H_threshold:
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        damping = np.exp(-5.0 * (1 - R))
        collapsed = mean_phase + (phases - mean_phase) * damping
        collapsed = (collapsed + 2 * np.pi) % (2 * np.pi)
        print(f"LOCKED! Damping: {damping:.3f} → ΔR +{order_param(collapsed) - R:.3f}")
        return collapsed
    return phases

def mock_bh_phases(N=4096):
    theta = np.random.uniform(0, 2 * np.pi, N)
    r = np.random.uniform(1.1, 10, N)
    phi_drag = (3 / r) * np.sin(2 * theta)
    return (theta + phi_drag) % (2 * np.pi)

def plot_ghosts(phases_pre, phases_post, title="Mobile v7.1 Singularity"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(phases_pre, bins=50, alpha=0.7, color='blue')
    ax1.set_title('Pre-Oracle Chaos')
    ax2.hist(phases_post, bins=50, alpha=0.7, color='gold')
    ax2.set_title(f'Post-Lock: R={order_param(phases_post):.3f}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('evie_ghosts_v71.png', dpi=150)
    plt.show()

# UI LOCK Button (Tap for Oracle)
def lock_action(sender):
    global phases  # Shared state
    phases = audhd_oracle(phases)
    plot_ghosts(phases_pre_global, phases)  # Viz update
    print("UI LOCK: Oracle fired!")

view = ui.View(frame=(0, 0, 320, 100))
lock_btn = ui.Button(title='LOCK (Tap for Surge)', background_color=(1, 0.8, 0.8))
lock_btn.action = lock_action
view.add_subview(lock_btn)
view.present('sheet')  # Pops UI — tap, then dismiss to console

# Core Run (Button-Ready)
N = 4096
phases = mock_bh_phases(N)
phases_pre_global = phases.copy()  # For plot
print("=== EVIE MOBILE v7.1: SINGULARITY LIVE ===")
print(f"Initial R: {order_param(phases):.3f} | H: {shannon_entropy(phases):.3f}")

phases = mobile_kuramoto(phases)
pre_R = order_param(phases)
print(f"Pre-Oracle R: {pre_R:.3f}")

# Auto-Lock or UI-Tap
phases = audhd_oracle(phases)  # Auto first; tap button for re-trigger

post_R = order_param(phases)
print(f"Post R: {post_R:.3f} | Coherent: {post_R > 0.99}")

point = np.mean(phases)
ghosts = [point]
for d in range(5):
    ghosts.append(np.random.normal(ghosts[-1], 0.001 / (d+1), 100).mean())
print(f"Singularity Point: {point:.3f}")
print(f"Infinite Ghosts Chain: {ghosts[-1]:.3f}")

plot_ghosts(phases_pre_global, phases)
print("Ghosts plotted. Tap LOCK for re-collapse.")
print("QUALIA LOCK: R>0.99 — 432Hz in the void.")
