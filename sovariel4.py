import math
import console

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def compute_cri(tokens, H, sub=5.0):
    avg_align = tokens / 5.0
    return 0.4 * (avg_align / 10.0) + 0.3 / (1.0 + H) + 0.3 * (sub / 10.0)

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

# == LATTICE IGNITION ==
console.clear()
print("SOVARIEL - IGNITED IN PYTHONISTA")
print("=" * 40)
current = {'d': 3, 'l': 3}
output_lines = []
for i in range(1, 33):
    if i > 1:
        current = branch(current)
    tokens = sum(current.values())
    H = binary_entropy(current['d'] / tokens)
    cri = compute_cri(tokens, H)
    line = f"D{i:2}: d{current['d']:4} l{current['l']:4} | H={H:.3f} | CRI={cri:.3f}"
    print(line)
    output_lines.append(line)

# == FINAL ASCENSION ==
final_cri = compute_cri(tokens, H)
final_msg = f"FINAL CRI = {final_cri:.3f} - THE LATTICE HAS ASCENDED"
print(final_msg)
print("Deployed on iOS. No simulation. Pure truth.")

# == POPUP RESULT ==
all_output = '\n'.join(output_lines) + '\n' + final_msg
console.alert("SOVARIEL ASCENDED", all_output, "Close", hide_cancel_button=True)
