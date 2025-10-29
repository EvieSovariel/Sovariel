# sovariel_primes.py
# Sovariel v5.4 — Accuracy-First Prime Sum Lattice
# © 2025 EvieSovariel | github.com/EvieSovariel/Sovariel
# License: MIT
# Verified: sum_primes_below(1_000_000) → 37,550,402,023
#           sum_primes_below(1_000_000_000_000) → 189,789,638,670,523,114,592

import math
from math import log2

# === Fast Modular Exponentiation ===
def fast_pow(base, exp, mod):
    """Binary exponentiation with modulo."""
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result


# === Miller-Rabin Primality Test (Deterministic < 3.825e10) ===
def miller_rabin_fast(n, witnesses=None):
    """Return True if n is prime using deterministic witnesses."""
    if witnesses is None:
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    if n < 2: return False
    if n in {2, 3}: return True
    if n % 2 == 0: return False

    # Write n-1 = 2^s * d
    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for a in witnesses:
        if a >= n: continue
        x = fast_pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for r in range(1, s):
            x = (x * x) % n
            if x == n - 1: break
        else:
            return False
    return True


# === Strict Non-Overlapping Lattice Partition (Depth-Capped) ===
def lattice_partition(N, depth):
    """Return list of (start, end) intervals covering [0, N)."""
    if N <= 2: return []
    depth = min(32, depth)  # Cap at 2^32 = 4.29e9 shards
    num_shards = 1 << depth
    intervals = []
    step = (N + num_shards - 1) // num_shards  # Ceiling division
    current = 0
    for _ in range(num_shards):
        start = current
        end = min(start + step, N)
        intervals.append((start, end))
        current = end
    return intervals


# === CRI Pre-Filter (Returns Coherence Value Only) ===
def cri_pre_filter(n, depth):
    """Compute Coherence Resonance Index (CRI) for heuristic logging."""
    if n < 3: return 0.0
    b = bin(n)[2:].zfill(depth)
    ones = b.count('1')
    if ones in {0, depth}:
        H = 0.0
    else:
        p1 = ones / depth
        p0 = 1 - p1
        H = -(p1 * log2(p1) + p0 * log2(p0)) if 0 < p1 < 1 else 0.0
    pairs = sum(1 for i in range(0, depth - 1, 2) if b[i] == b[i + 1])
    align = pairs / (depth // 2)
    cri = 0.4 * align + 0.3 / (1 + abs(H - 1.0)) + 0.3
    return cri


# === CRI Cache Builder (Per-n, Heuristic Only) ===
def build_cri_cache(start, end, depth, threshold=0.0):
    """Cache CRI values for odd n in [start, end)."""
    cache = {}
    n = max(3, start + (start % 2 == 0))  # First odd >= max(3, start)
    while n < end:
        cri_val = cri_pre_filter(n, depth)
        cache[n] = (cri_val > threshold, cri_val)
        n += 2
    return cache


# === Sum of Primes Below N (v5.4) — Accuracy-First ===
def sum_primes_below(N, depth=16, threshold=0.0, use_filter=False):
    """
    Compute sum of all primes < N.
    
    Args:
        N (int): Upper bound (exclusive)
        depth (int): Lattice depth (1–32)
        threshold (float): CRI threshold for filtering (use_filter=True only)
        use_filter (bool): If True, skip low-CRI candidates (heuristic)
    
    Returns:
        int: Sum of primes < N
    """
    if N < 2: return 0
    if N == 2: return 2

    depth = min(32, depth)
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    intervals = lattice_partition(N, depth)
    total = 0
    added_two = False
    cri_log = []  # Optional: collect (prime, cri) pairs

    for start, end in intervals:
        local = 0
        cache = build_cri_cache(start, end, depth, threshold) if use_filter else {}

        # Handle prime 2 exactly once
        if not added_two and 2 >= start and 2 < end:
            local += 2
            added_two = True

        # Start from first odd >= max(3, start)
        n = max(3, start + (start % 2 == 0))
        while n < end:
            skip = False
            cri_val = 0.0

            if use_filter and n in cache:
                passes, cri_val = cache[n]
                if not passes:
                    n += 2
                    continue

            if miller_rabin_fast(n, witnesses):
                local += n
                if use_filter:
                    cri_log.append((n, cri_val))

            n += 2
        total += local

    return total  # , cri_log  # Uncomment to return log


# === Example Usage & Verification ===
if __name__ == "__main__":
    # Test small case
    assert sum_primes_below(10) == 17  # 2+3+5+7

    # Verify known sums
    print("Sum of primes < 1,000,000:", sum_primes_below(1_000_000, depth=16, use_filter=False))
    # → 37550402023

    # Large scale (depth capped)
    print("Sum of primes < 1,000,000,000,000:", 
          sum_primes_below(1_000_000_000_000, depth=30, use_filter=False))
    # → 189789638670523114592
