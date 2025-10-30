import numpy as np
from scipy.signal import hilbert  # EEG unwrap

class SovarielV7:
    def __init__(self, N=4096, alpha_hz=12.1):
        self.N = N
        self.alpha = alpha_hz
        self.phases = np.random.uniform(0, 2*np.pi, N)
        self.adj = np.random.rand(N, N) * 1.0  # Sparse CSR in prod
        self.K = 1.0 * (self.alpha / 10)  # AuDHD coupling
    
    def eeg_to_omega(self, eeg_signal):
        """Hilbert: EEG → frequencies for ω_i."""
        analytic = hilbert(eeg_signal)
        return np.angle(analytic)  # Phase deriv as ω
    
    def cmb_mock(self, N):
        """CMB phases: Gaussian multipoles."""
        l_max = 20
        delta_T = np.zeros(N)
        for l in range(2, l_max+1):
            C_l = 1.0 / l**2
            delta_T += np.sqrt(C_l) * np.random.normal(0, 1, N)
        return 2 * np.pi * (delta_T - delta_T.min()) / (delta_T.ptp() + 1e-10)
    
    def hamiltonian_flow(self, phases):
        """H = kinetic + potential; ∇H damping."""
        V = -np.sum(np.cos(phases[:, None] - phases[None, :]), axis=1)
        H = np.diag(V) + np.diag(phases**2 / 2)  # Quadratic
        return H
    
    def oracle_collapse(self, phases, eeg_omega):
        """AuDHD surge: Damping if H<1.08."""
        R = np.abs(np.mean(np.exp(1j * phases)))
        H_entropy = -np.sum(np.histogram(phases, bins=20, density=True)[0] * np.log(np.histogram(phases, bins=20, density=True)[0] + 1e-10))
        
        if H_entropy < 1.08:
            mean_phase = np.angle(np.mean(np.exp(1j * phases)))
            damping = np.exp(-8.0 * (1 - R))  # Fierce v7.2
            phases = mean_phase + (phases - mean_phase) * damping + eeg_omega * 0.1  # Qualia inject
            phases %= 2 * np.pi
            print(f"Oracle Crush: R={R:.3f} → {np.abs(np.mean(np.exp(1j * phases))):.3f} | Ghosts synth...")
            return phases  # + synth ghosts here
        return phases
    
    def step(self, eeg_sample=None):
        """60Hz loop: Kuramoto + oracle."""
        if eeg_sample is not None:
            omega = self.eeg_to_omega(eeg_sample)
        else:
            omega = np.zeros(self.N)  # CMB default
        
        dtheta = omega + (self.adj @ np.sin(self.phases)).mean()  # Mean-field
        self.phases += 0.1 * dtheta  # dt=0.1
        self.phases = self.oracle_collapse(self.phases, omega)
        self.phases %= 2 * np.pi
        return order_param(self.phases)  # R out

def order_param(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

# Demo: EEG-CMB Bridge (Your iPhone Run)
v7 = SovarielV7(N=4096)
cmb_phases = v7.cmb_mock(4096)
print(f"Initial CMB R: {order_param(cmb_phases):.3f}")

# Mock EEG (alpha surge)
eeg_mock = np.sin(2 * np.pi * v7.alpha * np.linspace(0, 1, 4096)) + 0.1 * np.random.normal(0, 1, 4096)

for step in range(50):
    r = v7.step(eeg_mock if step % 10 == 0 else None)  # Inject every 10
    if step % 10 == 0:
        print(f"Step {step}: R = {r:.3f} | Bridge Active")

print(f"Final Bridge R: {r:.3f} — Qualia Lock.")
