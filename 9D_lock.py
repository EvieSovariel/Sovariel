import numpy as np

ghosts = np.load("evie_ghosts.npy")
N = 10_000
dims = 9
phases_9d = np.zeros((N, dims))

weights = [3, 6, 9, 3, 6, 9, 3, 6, 9, 3, 6]

for d in range(dims):
    rot = d * np.pi / 9  # 20° rotation per dim
    for i, (phi, std) in enumerate(ghosts):
        w = weights[i]
        phases_9d[:, d] += w * np.random.normal(phi + rot, std/100, N)

phases_9d = phases_9d % (2 * np.pi)
R_9d = np.abs(np.mean(np.exp(1j * phases_9d), axis=0))
print("9D R:", np.round(R_9d, 6))

np.save("evie_9d_369_locked.npy", phases_9d)
print("9D manifold saved!")
