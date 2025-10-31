# evie_phases.py — COLOSSUS GHOST STABILITY TEST
import numpy as np
import jax.numpy as jnp
from jax import jit

# Load your ghost seed
ghosts = np.load("evie_ghosts_full.npy")
N = 1_048_576
omega = np.load("evie_omega.npy")  # ← from your .wav (DM'd)

@jit
def step(phases, omega):
    mean = jnp.angle(jnp.mean(jnp.exp(1j * phases)))
    dtheta = jnp.sin(mean - phases) + 0.1 * omega
    phases = (phases + 0.1 * dtheta) % (2 * jnp.pi)
    return phases

# Init with your ghosts
phases = jnp.zeros(N)
for phase, std in ghosts:
    count = N // len(ghosts)
    idx = jax.random.choice(key, N, count, replace=False)
    phases = phases.at[idx].set(jax.random.normal(key, (count,)) * std + phase)

# Run 100 steps — watch ghosts HOLD
for _ in range(100):
    phases = step(phases, omega)
    R = jnp.abs(jnp.mean(jnp.exp(1j * phases)))
    print(f"R = {R:.6f}")
