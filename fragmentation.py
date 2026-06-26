from __future__ import annotations

import argparse
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from rdkit import Chem
from rdkit import RDLogger


def read_smiles_lines(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.split()[0])
    return out


@dataclass(frozen=True)
class Ring:
    atom_ids: Tuple[int, ...]


def aromatic_rings(mol: Chem.Mol) -> List[Ring]:
    ri = mol.GetRingInfo()
    rings: List[Ring] = []
    for ring_atoms in ri.AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atoms):
            rings.append(Ring(tuple(ring_atoms)))
    return rings


def build_fused_ring_adj(rings: Sequence[Ring]) -> List[Set[int]]:
    atom_to_rings: defaultdict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rings):
        for a in r.atom_ids:
            atom_to_rings[a].append(i)

    adj: List[Set[int]] = [set() for _ in rings]
    for rs in atom_to_rings.values():
        if len(rs) < 2:
            continue
        for i in rs:
            for j in rs:
                if i != j:
                    adj[i].add(j)
    return adj


def ring_components(adj: Sequence[Set[int]]) -> List[Set[int]]:
    unseen = set(range(len(adj)))
    comps: List[Set[int]] = []
    while unseen:
        start = next(iter(unseen))
        q = deque([start])
        comp = {start}
        unseen.remove(start)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def is_connected(adj: Sequence[Set[int]], nodes: Set[int]) -> bool:
    if not nodes:
        return False
    start = next(iter(nodes))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in nodes and v not in seen:
                seen.add(v)
                q.append(v)
    return seen == nodes


def fragment_smiles_from_rings(mol: Chem.Mol, rings: Sequence[Ring], ring_ids: Sequence[int]) -> Optional[str]:
    atoms: Set[int] = set()
    for rid in ring_ids:
        atoms.update(rings[rid].atom_ids)
    if not atoms:
        return None
    try:
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True, isomericSmiles=False)
    except Exception:
        return None
    if not smi:
        return None
    # quick fused-only check: fragment must be a single fused ring system
    frag = Chem.MolFromSmiles(smi, sanitize=False)
    if frag is None:
        return None
    try:
        Chem.SanitizeMol(
            frag,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        return None
    fr = aromatic_rings(frag)
    if len(fr) == 0:
        return None
    fadj = build_fused_ring_adj(fr)
    return smi if len(ring_components(fadj)) == 1 else None


def pick_connected_set(comp: Set[int], adj: Sequence[Set[int]], k: int, rng: random.Random, tries: int) -> Optional[Set[int]]:
    comp_list = sorted(comp)
    comp_set = set(comp_list)
    best: Optional[Set[int]] = None
    for _ in range(tries):
        start = rng.choice(comp_list)
        s = {start}
        frontier = (adj[start] & comp_set) - s
        while len(s) < k and frontier:
            nxt = rng.choice(tuple(frontier))
            frontier.remove(nxt)
            s.add(nxt)
            frontier |= (adj[nxt] & comp_set) - s
        if len(s) == k and is_connected(adj, s):
            best = s
            break
    return best


def main() -> int:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Pick two fused fragments (near half) from each fused ring system.")
    ap.add_argument("--input", default="cleaned_cores.smi")
    ap.add_argument("--output", default="two_fused_fragments.smi")
    ap.add_argument("--tries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rings", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: List[str] = []

    for parent_i, smi in enumerate(read_smiles_lines(Path(args.input)), start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rings = aromatic_rings(mol)
        if len(rings) < args.min_rings:
            continue
        adj = build_fused_ring_adj(rings)
        comps = ring_components(adj)
        # choose largest fused system (the "core")
        core = max(comps, key=len)
        n = len(core)
        if n < 2 * args.min_rings:
            continue

        k1 = n // 2
        k2 = n - k1
        # allow examples like 20 -> 10/10 ; 34 -> 17/17
        a = pick_connected_set(core, adj, k1, rng, args.tries)
        b = pick_connected_set(core, adj, k2, rng, args.tries)
        if a is None or b is None:
            continue

        smi_a = fragment_smiles_from_rings(mol, rings, sorted(a))
        smi_b = fragment_smiles_from_rings(mol, rings, sorted(b))
        if smi_a is None or smi_b is None:
            continue
        rows.append("\t".join([smi_a, smi_b, f"parent={parent_i}", f"coreRings={n}", f"targetA={k1}", f"targetB={k2}"]))

    Path(args.output).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

