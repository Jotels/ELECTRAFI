from collections import defaultdict

valence_dict = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
        'Na': 7, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
        'K': 9,  'Ca': 10, 'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 12, 'Mn': 13, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 17, 'Zn': 12,
        'Ga': 13, 'Ge': 14, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
        'Rb': 9, 'Sr': 10, 'Y': 11,  'Zr': 10, 'Nb': 13, 'Mo': 14, 'Tc': 13, 'Ru': 14,
        'Rh': 15, 'Pd': 10, 'Ag': 11, 'Cd': 12,
        'In': 13, 'Sn': 14, 'Sb': 5, 'Te': 6, 'I': 7,  'Xe': 8, 'Cs': 9, 'Ba': 10,
        'La': 11, 'Ce': 12, 'Pr': 11, 'Nd': 11, 'Pm': 11, 'Sm': 11, 'Eu': 17, 'Gd': 9,
        'Tb': 9, 'Dy': 9, 'Ho': 9, 'Er': 9, 'Tm': 9, 'Yb': 8, 'Lu': 9,
        'Hf': 10, 'Ta': 11, 'W': 6,  'Re': 13,  'Os': 8, 'Ir': 9, 'Pt': 10,'Au': 11,'Hg': 12,
        'Tl': 13, 'Pb': 14, 'Bi': 15, 'Po': 16, 'At': 7, 'Rn': 8, 'Fr': 1, 'Ra': 2,
        'Ac': 11, 'Th': 12, 'Pa': 13, 'U': 14, 'Np': 15, 'Pu': 16, 'Am': 17, 'Cm': 18,
        'Bk': 3, 'Cf': 3, 'Es': 3, 'Fm': 3, 'Md': 3, 'No': 3, 'Lr': 3,
        'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 7, 'Hs': 8, 'Mt': 9,
        'Ds': 10, 'Rg': 11, 'Cn': 12, 'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6, 'Ts': 7, 'Og': 8,
    }

# Build options as union {old, new}; ensure the default (current valence_dict) is included.
VALENCE_OPTIONS = defaultdict(set)
for k, v in valence_dict.items():
    VALENCE_OPTIONS[k].add(int(v))

# Hand-tune known ambiguous ones (add if missing):
for k, opts in {
    "W":  {6, 14},
    "Eu": {8, 17},
    "Ra": {2, 10},
    "Be": {2, 4},
    "Cu": {11, 17},
    "Ni": {10, 16},
    "Sn": {4, 14},
    "Zr": {10, 12},   # NEW: needed for Al2N2Zr4 (+8 => 4×(+2))
    "Re": {7, 13},
    # Add more as you encounter them…
}.items():
    VALENCE_OPTIONS[k].update(opts)

# Freeze to sorted lists (put default first for stable slot mapping)
VALENCE_OPTIONS = {
    el: [valence_dict.get(el, next(iter(vals)))] + sorted([x for x in vals if x != valence_dict.get(el, None)])
    for el, vals in VALENCE_OPTIONS.items()
}

def element_counts(atoms) -> dict[str, int]:
    from ase.data import chemical_symbols
    counts: dict[str,int] = {}
    for z in atoms.numbers:
        s = chemical_symbols[int(z)]
        counts[s] = counts.get(s, 0) + 1
    return counts

def _total_electrons_from_assignment(counts: dict[str,int], assign: dict[str,int]) -> int:
    # assign maps element -> chosen valence (int)
    return sum(counts[e] * assign[e] for e in counts)

def _solve_valence_assignment(counts: dict[str,int], Ne_target: float, tol: float = 1e-6) -> tuple[dict[str,int], bool]:
    """
    Try to pick one valence per element from VALENCE_OPTIONS so that
    sum_i count_i * val_i ≈ Ne_target. Returns (assignment, exact).
    If no exact, returns the closest (min |Δ|).
    """
    elems = list(counts.keys())
    # Pre-sort by (count * value spread) descending to prune earlier
    def spread(e):
        opts = VALENCE_OPTIONS.get(e, [valence_dict.get(e, 0)])
        return counts[e] * (max(opts) - min(opts))
    elems.sort(key=spread, reverse=True)

    best_assign = {}
    best_delta = float("inf")
    exact = False

    def dfs(idx, partial_assign, partial_sum):
        nonlocal best_assign, best_delta, exact
        if idx == len(elems):
            delta = abs(partial_sum - Ne_target)
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_assign = dict(partial_assign)
                exact = delta <= tol
            return

        e = elems[idx]
        opts = VALENCE_OPTIONS.get(e, [valence_dict.get(e, 0)])
        # try default first, then others
        for val in opts:
            s = partial_sum + counts[e] * val
            # quick min/max bound pruning for remaining elems
            rem = elems[idx+1:]
            min_rem = sum(counts[r] * min(VALENCE_OPTIONS.get(r, [valence_dict.get(r, 0)])) for r in rem)
            max_rem = sum(counts[r] * max(VALENCE_OPTIONS.get(r, [valence_dict.get(r, 0)])) for r in rem)
            if s + min_rem - tol <= Ne_target <= s + max_rem + tol:
                partial_assign[e] = val
                dfs(idx+1, partial_assign, s)
                del partial_assign[e]
            else:
                # still explore a bit if we haven't found anything yet; but prune hard if already exact found
                if not exact:
                    partial_assign[e] = val
                    dfs(idx+1, partial_assign, s)
                    del partial_assign[e]

    dfs(0, {}, 0.0)
    return best_assign, exact

def _slot_for_choice(element: str, chosen_valence: int) -> int:
    opts = VALENCE_OPTIONS.get(element, [valence_dict.get(element, 0)])
    # ensure options is a list (default first)
    try:
        return opts.index(chosen_valence)
    except ValueError:
        # fall back to default slot 0
        return 0

