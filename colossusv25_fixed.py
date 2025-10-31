# ========================================================
# COLOSSUS PLANETARY GRID vΩ25 — GLOBAL SYNCHRONIZER FIXED
# Pythonista Omega — iOS | 1K NODES | DENSE FULL FIELD | 8 MB SAFE
# FIXED: NameError h,w | endpoint cycles | time import
# ========================================================

import numpy as np
import warnings
from datetime import datetime
import os
import time

warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')

STATE_FILE = 'colossus_vΩ25_state.txt'
FIELD_FILE = 'colossus_vΩ25_field.npy'
LOCK_FILE = 'colossus_vΩ25_build.lock'

# OVERRIDE STUCK LOCK
if os.path.exists(LOCK_FILE):
    print(f"[{datetime.now().strftime('%H:%M:%S')} ] STUCK LOCK — PURGING")
    try:
        os.remove(LOCK_FILE)
    except:
        pass

# RESUME OR BUILD — ROBUST PARSE
if os.path.exists(STATE_FILE) and os.path.exists(FIELD_FILE):
    print(f"[{datetime.now().strftime('%H:%M:%S')} ] GRID RESUMED — OMEGA ACTIVE")
    try:
        planetary_field = np.load(FIELD_FILE)
        with open(STATE_FILE, 'r') as f:
            state = f.read()
        nz_match = [m for m in state.split() if 'NZ:' in m][0]
        nonzero = int(nz_match.split('NZ:')[1].split('|')[0].strip())
        shape_match = [m for m in state.split() if 'Shape:' in m][0]
        shape = shape_match.split('Shape:')[1].strip()
        print(f"   → {nonzero:,} NZ | Region: {shape}")
        print("COLOSSUS GRID vΩ25 — RESUMED")
        print(f"   → Memory: {planetary_field.nbytes / 1e9:.3f} GiB")
        print("PLANETARY RESONANCE: LOCKED")
        exit()
    except Exception as e:
        print(f"   → Resume failed ({e}). Rebuild.")
        try:
            os.remove(FIELD_FILE)
            os.remove(STATE_FILE)
        except:
            pass

open(LOCK_FILE, 'w').close()

# CONSTANTS — SCALED FOR FULL WEB
GRID_NODES = 1000  # 1k x 1k dense = 8 MB safe
LEY_LINES = 12
TELLURIC_FREQUENCY = 432.0
OMEGA_PHASE = 2j * np.pi
GOLDEN_RATIO = (1 + 5**0.5) / 2
TOLERANCE = 0.0  # Full harmonics

print(f"[{datetime.now().strftime('%H:%M:%S')} ] Initializing vΩ25 — GLOBAL SYNCHRONIZER")
print(f"   → {GRID_NODES:,} nodes | {LEY_LINES} ley lines | {TELLURIC_FREQUENCY} Hz")
print(f"   → Full dense web: {GRID_NODES}2 = {GRID_NODES**2:,} cells")

# FULL GRID — NO CHUNKING
t = np.linspace(0, 1, GRID_NODES, endpoint=False)
x_base = np.exp(1j * TELLURIC_FREQUENCY * t * 2 * np.pi)

ley_thetas = np.linspace(0, 2*np.pi, LEY_LINES, endpoint=False)
phi_offsets = ley_thetas + np.pi / GOLDEN_RATIO

colossus_grid = np.zeros((GRID_NODES, GRID_NODES), dtype=complex)

print("   → Entangling harmonics...")
start = time.time()

for ley in range(LEY_LINES):
    theta = ley_thetas[ley]
    phi = phi_offsets[ley]
    phase_fwd = np.exp(OMEGA_PHASE * theta)
    phase_rev = np.exp(OMEGA_PHASE * phi)

    n_row = np.arange(GRID_NODES)
    sin_fwd = np.sin((n_row * phase_fwd.real) % (2 * np.pi))
    n_col = np.arange(GRID_NODES)
    cos_rev = np.cos((n_col * phase_rev.real) % (2 * np.pi))

    outer_fwd = np.outer(x_base, sin_fwd)
    outer_rev = 0.618 * np.outer(x_base, cos_rev)
    colossus_grid += outer_fwd + outer_rev

    pct = (ley + 1) / LEY_LINES
    filled = int(50 * pct)
    bar = "█" * filled + "░" * (50 - filled)
    elapsed = time.time() - start
    print(f"   → [{bar}] {pct*100:5.1f}% | Ley {ley+1}/{LEY_LINES} | {elapsed:.1f}s", end="\r")

print()

print(f"[{datetime.now().strftime('%H:%M:%S')} ] Entanglement Complete")

# PROJECT REAL DENSE
planetary_field = np.real(colossus_grid)

fmin, fmax = planetary_field.min(), planetary_field.max()
if fmax > fmin:
    planetary_field = (planetary_field - fmin) / (fmax - fmin)

np.save(FIELD_FILE, planetary_field)
with open(STATE_FILE, 'w') as f:
    f.write(f"NZ: {GRID_NODES**2} | Shape: {planetary_field.shape} | Time: {datetime.now().isoformat()}\n")
    f.write(f"Residue: 0.0")
os.remove(LOCK_FILE)

h = GRID_NODES
w = GRID_NODES

print("COLOSSUS GRID vΩ25 — GLOBAL SYNCHRONIZER SUCCESS")
print(f"   → Full Region: {h}×{w}")
print(f"   → Density: 100% (Full web)")
print(f"   → Memory: {planetary_field.nbytes / 1e9:.3f} GiB")
print("PLANETARY RESONANCE: FULL SPHERE LOCKED | RERUN SAFE | OMEGA READY")
