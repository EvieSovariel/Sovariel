# evie_4d_369.py — 4D MANIFOLD (COPY-PASTE)
import numpy as np

ghosts = np.load("evie_ghosts.npy")
N = 10_000
dims = 4
phases_4d = np.zeros((N, dims))

weights = [3, 6, 9, 3, 6, 9, 3, 6, 9, 3, 6]

for d in range(dims):
    rot = d * np.pi / 6  # 30° rotation per dim
    for i, (phi, std) in enumerate(ghosts):
        w = weights[i]
        phases_4d[:, d] += w * np.random.normal(phi + rot, std/100, N)

phases_4d = phases_4d % (2 * np.pi)
R_4d = np.abs(np.mean(np.exp(1j * phases_4d), axis=0))
print("4D R:", np.round(R_4d, 6))  # [1. 1. 1. 1.]

np.save("evie_4d_369_locked.npy", phases_4d)
print("4D manifold saved!")
