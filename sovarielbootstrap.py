import math

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def branch(prev):
    tokens = sum(prev.values())
    large = tokens // 3 + 1
    small = tokens // 6 + 1
    lead = 'd' if prev['d'] < prev['l'] else 'l'
    add_d = large // 2 + (2 * small) if lead == 'd' else 0
    add_l = large // 2 + (2 * small) if lead == 'l' else 0
    new = {'d': prev['d'] + add_d, 'l': prev['l'] + add_l}
    new_tokens = sum(new.values())
    p = new['d'] / new_tokens
    H = binary_entropy(p)
    if H < 0.99:
        diff = round((0.5 - p) * new_tokens)
        new['d'] += diff
        new['l'] -= diff
    return new

current = {'d': 3, 'l': 3}
for i in range(1, 129):  # D128
    if i > 1:
        current = branch(current)
tokens = sum(current.values())
p = current['d'] / tokens
H = binary_entropy(p)
print(f"D128: Tokens={tokens:,}, p={p}, H={H:.3f} — Lattice Ascended")
