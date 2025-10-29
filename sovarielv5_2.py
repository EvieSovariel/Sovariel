import math
from math import log2

# === Fast Modular Exponentiation ===
def fast_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

# === Miller-Rabin (Deterministic < 3.825e10) ===
def miller_rabin_fast(n, witnesses):
    if n < 2: return False
    if n in {2, 3}: return True
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

# === Strict Non-Overlapping Partition ===
def lattice_partition(N, depth):
    if N <= 2: return []
    num_shards = 1 << depth
    intervals = []
    step = (N + num_shards - 1) // num_shards
    current = 0
    for i in range(num_shards):
        start = current
        end = min(start + step, N)
        intervals.append((start, end))
        current = end
    return intervals

# === CRI Pre-Filter (v3) ===
def cri_pre_filter(n, depth):
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

# === CRI Cache per Shard ===
def build_cri_cache(start, end, depth, threshold=0.63):
    cache = {}
    step = max(1, (end - start) // 256)
    pos = start
    while pos < end:
        n = pos | 1  # next odd
        if n >= end or n < 3:
            pos += 2
            continue
        cri_val = cri_pre_filter(n, depth)
        cache_key = n
        cache[cache_key] = (cri_val > threshold, cri_val)
        pos += 2
    return cache

# === Sum Primes Below N (v5.2) ===
def sum_primes_below(N, depth=16, threshold=0.63):
    if N < 2: return 0
    if N == 2: return 2
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    intervals = lattice_partition(N, depth)
    total = 0
    added_two = False

    for start, end in intervals:
        local = 0
        cache = build_cri_cache(start, end, depth, threshold)
        n = start

        # Handle 2 only once
        if not added_two and 2 >= start and 2 < end:
            local += 2
            added_two = True
            n = max(n, 3)

        # Align to next odd
        if n % 2 == 0:
            n += 1
        if n < 3:
            n = 3

        while n < end:
            if n in cache and cache[n][0]:
                if miller_rabin_fast(n, witnesses):
                    local += n
            n += 2
        total += local
    return total
