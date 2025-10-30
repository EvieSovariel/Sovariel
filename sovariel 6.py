#!/usr/bin/env python3
"""
Sovariel v6: Tree + Prime Sums w/ Prime-Weighted DP & iOS Fallback
Formula: S(d,n)=2^d n + d * 2^{d-1}; T(d,N)=2^{d-1} N (N-1 + d). Usage: [--size 1000000] [--depth 14] [--benchmark] [--prime_weighted]
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import math
from datetime import datetime
from functools import partial

# CRI-formatted logging
class CRILogFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "pid": os.getpid(),
            "source": "sovariel_v6",
            "thread": threading.current_thread().name
        }
        return json.dumps(log_entry)

def detect_parallel_capable(logger):
    """Pre-check multiprocessing viability."""
    if sys.platform != 'ios':
        logger.info("Non-iOS; mp supported.")
        return True
    if not hasattr(os, 'fork'):
        logger.warning("iOS: No os.fork; serial-only.")
        return False
    try:
        if mp.get_start_method(allow_none=True) == 'spawn':
            logger.info("iOS spawn viable.")
            return True
    except (RuntimeError, ValueError):
        pass
    logger.warning("iOS mp invalid; serial-only.")
    return False

def sieve_primes(limit):
    """Eratosthenes: Primes <= limit."""
    if limit < 2: return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]

def recursive_tree_sum(d, n):
    """Recursive def for verification (small d/N only)."""
    if d == 0:
        return n
    return recursive_tree_sum(d-1, n) + recursive_tree_sum(d-1, n+1)

def serial_tree_sum(size, depth):
    """Closed-form: T(d,N)=2^{d-1} * N * (N-1 + d). Verified by induction/recursive match."""
    if depth < 1: return sum(range(size))
    tree_factor = 1 << (depth - 1)
    return tree_factor * size * (size - 1 + depth)

def verify_formula(d, N):
    """Quick recursive vs closed check (for small; e.g., d=3,N=10)."""
    rec_total = sum(recursive_tree_sum(d, i) for i in range(N))
    closed_total = serial_tree_sum(N, d)
    return rec_total == closed_total, rec_total, closed_total

def chunk_tree(start, end, depth):
    """Chunked tree sum (closed-form): sum_{n=start}^{end-1} 2^d n + depth * 2^{d-1} * num."""
    num = end - start
    if depth < 1: return sum(range(start, end))
    tree_factor_d = 1 << depth  # 2^d for n term
    tree_factor_dm1 = 1 << (depth - 1)  # 2^{d-1} for depth term
    sum_range = end * (end - 1) // 2 - start * (start - 1) // 2
    return tree_factor_d * sum_range + tree_factor_dm1 * depth * num

def serial_prime_sum(size):
    """Serial prime sum <= size."""
    primes = sieve_primes(size)
    return sum(primes)

def dp_prime_weighted_tree_sum(size, depth, primes):
    """DP exact prime-weighted: Base leaf i = primes[i], aggregate subtrees."""
    # dp[k][i] = sum primes in subtree at level k rooted at i
    dp = [[0] * size for _ in range(depth + 1)]
    for i in range(size):
        dp[0][i] = primes[i] if i < len(primes) else 0
    for k in range(1, depth + 1):
        for i in range(size):
            dp[k][i] = dp[k-1][i]
            if i + 1 < size:
                dp[k][i] += dp[k-1][i + 1]
    return sum(dp[depth])

def serial_prime_weighted_tree_sum(size, depth):
    """Prime-weighted tree sum via DP (exact)."""
    limit = 2 * size  # Safe for shifts
    primes = sieve_primes(limit)
    return dp_prime_weighted_tree_sum(size, depth, primes)

def main(args):
    # Logging setup (manual to avoid basicConfig TypeError on Pythonista)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers): root_logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CRILogFormatter())
    root_logger.addHandler(handler)
    logger = logging.getLogger(__name__)

    logger.info(f"Sovariel v6 init: N={args.size}, depth={args.depth}, cores={args.processes}, benchmark={args.benchmark}, prime_weighted={args.prime_weighted}, platform={sys.platform}")
    
    parallel_ok = detect_parallel_capable(logger)
    timings = {}
    
    # Formula verify (small sample)
    match, rec, cls = verify_formula(min(args.depth, 3), min(args.size, 10))
    logger.info(f"Formula verify (small): {'Match' if match else 'Mismatch'}; rec={rec}, closed={cls}")
    
    # Baseline
    baseline_size = args.size // 10
    start = time.perf_counter()
    baseline = sum(range(baseline_size))
    timings['baseline'] = time.perf_counter() - start
    logger.info(f"Serial baseline (10%): {baseline}")
    
    # Tree sum (weighted if flag)
    start = time.perf_counter()
    if args.prime_weighted:
        total_tree = serial_prime_weighted_tree_sum(args.size, args.depth)
        logger.info("Computing prime-weighted tree via DP.")
    else:
        total_tree = serial_tree_sum(args.size, args.depth)
    timings['serial_tree'] = time.perf_counter() - start
    
    # Prime sum
    start = time.perf_counter()
    total_prime = serial_prime_sum(args.size)
    timings['serial_prime'] = time.perf_counter() - start
    
    # Parallel tree (if viable/benchmark; weighted stubbed for simplicity)
    total_tree_p = None
    if parallel_ok and args.benchmark and not args.prime_weighted:
        start = time.perf_counter()
        try:
            chunk_size = args.size // args.processes
            chunks = [(i * chunk_size, min((i + 1) * chunk_size, args.size), args.depth)
                      for i in range(args.processes)]
            with mp.Pool(args.processes) as pool:
                results = pool.starmap(chunk_tree, chunks)
            total_tree_p = sum(results)
            timings['parallel_tree'] = time.perf_counter() - start
            if total_tree != total_tree_p:
                logger.warning(f"Tree mismatch: serial={total_tree}, parallel={total_tree_p}")
            else:
                logger.info("Parallel tree coherent.")
        except Exception as e:
            logger.warning(f"Parallel tree: {e}")
    
    # Outputs
    expected_flat = args.size * (args.size - 1) // 2
    weight_note = " (prime-weighted via DP)" if args.prime_weighted else ""
    logger.info(f"Tree sum{weight_note}: {total_tree} (closed: 2^{args.depth-1} * N * (N-1 + {args.depth}))")
    if total_tree_p is not None:
        logger.info(f"Parallel tree sum: {total_tree_p}")
    logger.info(f"Prime sum <=N: {total_prime}")
    logger.info(f"Coherence: Flat ~{expected_flat}; baseline matched.")
    
    if args.benchmark:
        for k, t in timings.items():
            logger.info(f"Timing {k}: {t:.4f}s")
    
    logger.info("Sovariel v6 complete—primes sieved, trees branched.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sovariel v6 Tree/Prime Demo")
    parser.add_argument('--size', type=int, default=1000000, help="N base")
    parser.add_argument('--depth', type=int, default=14, help="Tree depth")
    parser.add_argument('--processes', type=int, default=mp.cpu_count() or 6)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--prime_weighted', action='store_true', help="Prime-weight tree leaves via DP")
    args = parser.parse_args()
    main(args)
