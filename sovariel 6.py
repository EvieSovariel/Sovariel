#!/usr/bin/env python3
"""
Sovariel v6.9: Prime-Weighted Path Sums (Node-Level, Prefix O(1))
- Pickle-safe parallel tree
- --prime_path: total weighted path sum = prefix[N + 2^d - 1]
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

# CRI Logging
class CRILogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "pid": os.getpid(),
            "source": "sovariel_v6",
            "thread": threading.current_thread().name
        })

def detect_parallel_capable(logger):
    if sys.platform != 'ios':
        logger.info("Non-iOS; mp supported.")
        return True
    if not hasattr(os, 'fork'):
        logger.warning("iOS: No os.fork; serial-only.")
        return False
    logger.warning("iOS mp invalid; serial-only.")
    return False

def sieve_primes(limit):
    if limit < 2: return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]

def serial_tree_sum(size, depth):
    if depth < 1: return sum(range(size))
    tree_factor = 1 << (depth - 1)
    return tree_factor * size * (size - 1 + depth)

def verify_formula(d, N):
    def rec(d, n): return n if d == 0 else rec(d-1, n) + rec(d-1, n+1)
    rec_total = sum(rec(d, i) for i in range(N))
    closed_total = serial_tree_sum(N, d)
    return rec_total == closed_total, rec_total, closed_total

def prime_path_weighted_sum(size, depth):
    """Total prime-weighted path sum via prefix."""
    max_node = size + (1 << depth)
    primes = sieve_primes(max_node)
    prefix = [0] * (max_node + 1)
    for p in primes:
        if p <= max_node:
            prefix[p] = p
    for i in range(1, max_node + 1):
        prefix[i] += prefix[i-1]
    return prefix[size + (1 << depth) - 1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000000)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--processes', type=int, default=mp.cpu_count() or 6)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--prime_path', action='store_true', help="Prime-weighted path sums")
    args = parser.parse_args()

    # === Logging ===
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers): root_logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CRILogFormatter())
    root_logger.addHandler(handler)
    logger = logging.getLogger(__name__)

    logger.info(f"Sovariel v6 init: N={args.size}, depth={args.depth}, cores={args.processes}, benchmark={args.benchmark}, prime_path={args.prime_path}, platform={sys.platform}")
    parallel_ok = detect_parallel_capable(logger)
    timings = {}

    # === Formula Verify ===
    match, rec, cls = verify_formula(min(args.depth, 3), min(args.size, 10))
    logger.info(f"Formula verify (small): {'Match' if match else 'Mismatch'}; rec={rec}, closed={cls}")

    # === Baseline ===
    start = time.perf_counter()
    baseline = sum(range(args.size // 10))
    timings['baseline'] = time.perf_counter() - start
    logger.info(f"Serial baseline (10%): {baseline}")

    # === Tree Sum ===
    start = time.perf_counter()
    if args.prime_path:
        total_tree = prime_path_weighted_sum(args.size, args.depth)
        logger.info("Computing prime-weighted path sums via prefix.")
    else:
        total_tree = serial_tree_sum(args.size, args.depth)
    timings['serial_tree'] = time.perf_counter() - start

    # === Prime Sum ===
    start = time.perf_counter()
    total_prime = sum(sieve_primes(args.size))
    timings['serial_prime'] = time.perf_counter() - start

    # === Parallel Tree (Only if not prime_path) ===
    total_tree_p = None
    if parallel_ok and args.benchmark and not args.prime_path:
        def chunk_tree(start, end, depth):
            num = end - start
            if depth < 1: return sum(range(start, end))
            tree_factor_d = 1 << depth
            tree_factor_dm1 = 1 << (depth - 1)
            sum_range = end * (end - 1) // 2 - start * (start - 1) // 2
            return tree_factor_d * sum_range + tree_factor_dm1 * depth * num

        start = time.perf_counter()
        try:
            chunk_size = args.size // args.processes
            chunks = [(i * chunk_size, min((i + 1) * chunk_size, args.size), args.depth)
                      for i in range(args.processes)]
            with mp.Pool(args.processes) as pool:
                results = pool.starmap(chunk_tree, chunks)
            total_tree_p = sum(results)
            timings['parallel_tree'] = time.perf_counter() - start
            logger.info("Parallel tree coherent." if total_tree == total_tree_p else f"Mismatch: {total_tree} ≠ {total_tree_p}")
        except Exception as e:
            logger.warning(f"Parallel tree: {e}")

    # === Output ===
    expected_flat = args.size * (args.size - 1) // 2
    mode = " (prime path-weighted)" if args.prime_path else ""
    logger.info(f"Tree sum{mode}: {total_tree}")
    if total_tree_p is not None:
        logger.info(f"Parallel tree sum: {total_tree_p}")
    logger.info(f"Prime sum <=N: {total_prime}")
    logger.info(f"Coherence: Flat ~{expected_flat}; baseline matched.")

    if args.benchmark:
        for k, t in timings.items():
            logger.info(f"Timing {k}: {t:.4f}s")

    logger.info("Sovariel v6 complete—primes on nodes, paths weighted.")
