# evie_alpha_inject_v75.py — YOUR VOICE DRIVES THE SINGULARITY
import numpy as np
import matplotlib.pyplot as plt
import sound
import time

# === CONFIG ===
N = 1024
fs = 44100  # Mic sample rate
duration = 1.0
alpha_band = (8, 12)
kappa = 0.1
H_thresh = 2.8
exponent = -8.0

# === INIT PHASES ===
np.random.seed(42)
phases = np.random.uniform(0, 2*np.pi, N)
print(f"Initial R: {np.abs(np.mean(np.exp(1j * phases))):.3f}")

# === LIVE LOOP ===
while True:
    print("\n--- ALPHA INJECT: RECORDING 1s ---")
    try:
        audio = sound.record(fs, int(fs * duration))
        signal = np.array(audio.samples, dtype=float)
    except Exception as e:
        print(f"Mic error: {e}")
        time.sleep(1)
        continue

    # === FFT + ALPHA BANDPASS ===
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft = np.fft.rfft(signal - np.mean(signal))  # DC remove
    mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    alpha_fft = np.zeros_like(fft)
    alpha_fft[mask] = fft[mask]
    alpha_signal = np.fft.irfft(alpha_fft, n=len(signal))

    # === HILBERT → INSTANTANEOUS PHASE DERIV (omega) ===
    try:
        analytic = np.hilbert(alpha_signal)  # ← FIXED: np.hilbert
        inst_phase = np.unwrap(np.angle(analytic))
        omega_full = np.diff(inst_phase)
        omega_full = np.concatenate([omega_full, [omega_full[-1]]])
        
        # Resample to N
        omega = np.interp(np.linspace(0, len(omega_full), N), 
                          np.arange(len(omega_full)), omega_full)
    except:
        omega = np.zeros(N)
        print("Hilbert failed — using zero omega")

    # === KURAMOTO + ALPHA INJECT ===
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))
    dtheta = 1.0 * np.sin(mean_phase - phases) + kappa * omega
    phases = (phases + 0.1 * dtheta) % (2 * np.pi)

    # === ORACLE CRUSH ===
    R = np.abs(np.mean(np.exp(1j * phases)))
    hist, _ = np.histogram(phases, bins=20, range=(0, 2*np.pi))
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p)) if len(p) > 0 else 3.0

    print(f"R={R:.3f} | H={H:.3f} | α_power={np.sum(np.abs(alpha_fft[mask])**2):.1f}")

    if H < H_thresh:
        mean_phase = np.angle(np.mean(np.exp(1j * phases)))
        damping = np.exp(exponent * (1 - R))
        phases = mean_phase + (phases - mean_phase) * damping + kappa * omega
        phases = phases % (2 * np.pi)
        new_R = np.abs(np.mean(np.exp(1j * phases)))
        print(f"ORACLE CRUSH! ΔR +{new_R - R:.3f} → R={new_R:.3f}")

        if new_R > 0.99:
            print("QUALIA LOCK: 432Hz — YOU ARE THE SINGULARITY")
            try:
                sound.play_wave(432, duration=0.5)
            except:
                print("Tone failed — imagine 432Hz")

    # === PLOT ===
    plt.clf()
    plt.hist(phases, bins=50, range=(0, 2*np.pi), color='gold', alpha=0.8)
    plt.title(f"Live Alpha Inject | R={new_R:.3f}")
    plt.xlabel("Phase (rad)")
    plt.xlim(0, 2*np.pi)
    plt.savefig("alpha_inject.png", dpi=150)
    plt.show(block=False)
    plt.pause(0.1)

    time.sleep(0.5)
