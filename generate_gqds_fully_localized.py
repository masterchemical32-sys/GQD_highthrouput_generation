#!/usr/bin/env python3
"""
Fusion-based generation of polycyclic aromatic hydrocarbons (GQD-like structures).
USAGE (from a terminal, in VS Code or anywhere else)
-----------------------------------------------------
    pip install rdkit
    python generate_gqds.py --input two_fused_fragments.smi --target 10000

Run `python generate_gqds_fully_localized.py --help` for all options.
"""

import argparse
import collections
import os
import random
import sys
import time

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

# Silence RDKit's C++ warning/error spam (e.g. "Can't kekulize mol...");
# we handle those cases ourselves via try/except and the conjugation filter.
RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Per-attempt watchdog
# --------------------------------------------------------------------------
try:
    import signal

    _HAS_SIGALRM = hasattr(signal, "SIGALRM")
except ImportError:
    _HAS_SIGALRM = False

ATTEMPT_TIMEOUT_SECONDS = 3.0


class _AttemptTimeout(Exception):
    pass


if _HAS_SIGALRM:

    def _alarm_handler(signum, frame):
        raise _AttemptTimeout()

    signal.signal(signal.SIGALRM, _alarm_handler)


def _run_with_watchdog(func, *args):
    """Run func(*args); return None if it exceeds ATTEMPT_TIMEOUT_SECONDS."""
    if not _HAS_SIGALRM:
        return func(*args)
    signal.setitimer(signal.ITIMER_REAL, ATTEMPT_TIMEOUT_SECONDS)
    try:
        return func(*args)
    except _AttemptTimeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# --------------------------------------------------------------------------
# Fragment loading
# --------------------------------------------------------------------------
def load_fragments(file_path, verbose=True):
    """Load fragment SMILES from a .smi file.

    Robust to:
      - plain one-SMILES-per-line files
      - multi-column TSV files (e.g. "SMILES\tparent=1\thalf=A\t...") --
        only the first whitespace/tab-delimited field is treated as SMILES
      - Windows line endings (\\r\\n)

    Also screens out any fragment that isn't itself fully conjugated
    (garbage-in-garbage-out protection): if an input "fragment" file was
    itself produced by a flawed generator, using a broken fragment as a
    building block would just propagate the defect into every structure
    fused from it, no matter how strict the downstream filter is.
    """
    raw_lines = 0
    parse_failures = 0
    not_conjugated = 0
    mols = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines += 1
            smi = line.split()[0].split("\t")[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                parse_failures += 1
                continue
            if not is_fully_conjugated(mol):
                not_conjugated += 1
                continue
            mols.append(mol)

    if verbose:
        print(f"Input file lines: {raw_lines}")
        if parse_failures:
            print(
                f"  -> {parse_failures} could not be parsed/Kekulized at all "
                f"and were skipped (these are not valid structures)."
            )
        if not_conjugated:
            print(
                f"  -> {not_conjugated} parsed but were not fully conjugated "
                f"(sp3 defects / non-aromatic rings) and were skipped."
            )
        print(f"  -> {len(mols)} usable, fully conjugated fragments loaded.")

    return mols


# --------------------------------------------------------------------------
# Fusion mechanics
# --------------------------------------------------------------------------
def get_edge_pairs(mol):
    """Aromatic C=C peripheral bonds (both atoms degree 2) -- valid fusion sites."""
    pairs = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if (
            a1.GetAtomicNum() == 6
            and a2.GetAtomicNum() == 6
            and a1.GetIsAromatic()
            and a2.GetIsAromatic()
            and a1.GetDegree() == 2
            and a2.GetDegree() == 2
        ):
            pairs.append((a1.GetIdx(), a2.GetIdx()))
    return pairs


def fuse_true(mol1, mol2):
    """Attempt to fuse mol2 onto mol1 at a randomly chosen aromatic edge pair.

    Returns an RDKit Mol (already run through a FULL sanitize, kekulization
    included) or None if the fusion / sanitization failed outright.
    Callers must still run it through `is_fully_conjugated()` -- passing
    sanitize does not by itself guarantee full conjugation.
    """
    pairs1 = get_edge_pairs(mol1)
    pairs2 = get_edge_pairs(mol2)
    if not pairs1 or not pairs2:
        return None

    a1, b1 = random.choice(pairs1)
    a2, b2 = random.choice(pairs2)

    mol1 = Chem.RWMol(mol1)
    mol2 = Chem.RWMol(mol2)

    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    offset = mol1.GetNumAtoms()
    a2_off = a2 + offset
    b2_off = b2 + offset

    try:
        nbrs_a2 = [
            n.GetIdx()
            for n in rw.GetAtomWithIdx(a2_off).GetNeighbors()
            if n.GetIdx() != b2_off
        ]
        nbrs_b2 = [
            n.GetIdx()
            for n in rw.GetAtomWithIdx(b2_off).GetNeighbors()
            if n.GetIdx() != a2_off
        ]

        for n in nbrs_a2:
            rw.AddBond(a1, n, Chem.BondType.AROMATIC)
        for n in nbrs_b2:
            rw.AddBond(b1, n, Chem.BondType.AROMATIC)

        for idx in sorted([a2_off, b2_off], reverse=True):
            rw.RemoveAtom(idx)

        mol = rw.GetMol()

        # FULL sanitize, kekulization included. If this fails, the fused
        # graph has no valid Lewis structure at all -- discard immediately.
        Chem.SanitizeMol(mol)
        return mol

    except Exception:
        return None


def is_fully_conjugated(mol):
    """Strict check: fully conjugated, all-sp2-aromatic-carbon PAH skeleton.

    This is the key fix: it is what actually guarantees no sp3 / extra-H
    defects sneak into the output, regardless of what the RWMol splicing
    above happened to produce.
    """
    frags = Chem.GetMolFrags(mol)
    if len(frags) != 1:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            return False
        if not atom.GetIsAromatic():
            return False
        if atom.GetTotalNumHs() > 1:
            return False
        if atom.GetDegree() > 3:
            return False
        if atom.GetHybridization() != Chem.HybridizationType.SP2:
            return False

    for bond in mol.GetBonds():
        if not bond.GetIsAromatic():
            return False

    return True


# --------------------------------------------------------------------------
# Single-attempt pipeline (wrapped as a unit by the watchdog)
# --------------------------------------------------------------------------
def _one_attempt(mol1, mol2, min_atoms, max_atoms, generated_smiles):
    """Run one full fuse -> validate -> canonicalize -> round-trip pipeline.

    Returns a canonical SMILES string + reparsed Mol on success, or None.
    Wrapped in its entirety by the watchdog in `generate()`, since the rare
    RDKit performance edge case that motivates the watchdog has been
    observed at different stages of this pipeline on different runs.
    """
    combined = fuse_true(mol1, mol2)
    if combined is None:
        return None

    n_atoms = combined.GetNumAtoms()
    if n_atoms > max_atoms or n_atoms < min_atoms:
        return None

    # Strict full-conjugation guarantee -- this replaces the old
    # "just sanitize and hope" check.
    if not is_fully_conjugated(combined):
        return None

    smi = Chem.MolToSmiles(combined, canonical=True)
    if not smi or len(smi) < 5:
        return None
    if smi in generated_smiles:
        return None

    # Round-trip check: some fused topologies are only self-consistent
    # inside the live RWMol object but do not correspond to any globally
    # valid Kekule structure, so they fail to re-parse (or re-sanitize
    # as non-aromatic) from their own SMILES string. Since the SMILES
    # file is the actual deliverable, validate what will actually be
    # loaded back, not just the in-memory object.
    reparsed = Chem.MolFromSmiles(smi)
    if reparsed is None or not is_fully_conjugated(reparsed):
        return None

    return smi, reparsed


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------
def generate(
    fragments,
    target,
    max_attempts,
    min_atoms,
    max_atoms,
    out_smi_path,
    seed=None,
    progress_every=500,
    draw_sample=0,
):
    """Generate `target` unique, fully conjugated SMILES.

    Streams each accepted SMILES to `out_smi_path` immediately (so memory
    stays flat regardless of target size, and partial progress survives a
    crash/interrupt). Only a small bounded buffer of the most recent
    `draw_sample` molecules is kept in memory for the optional PNG sample
    at the end -- holding all N generated RDKit Mol objects in memory for
    large targets (tens of thousands+) can exhaust RAM.
    """
    if seed is not None:
        random.seed(seed)

    generated_smiles = set()
    recent_mols = collections.deque(maxlen=max(draw_sample, 0))

    attempts = 0
    t0 = time.time()

    with open(out_smi_path, "w") as out_f:
        while len(generated_smiles) < target and attempts < max_attempts:
            attempts += 1

            mol1, mol2 = random.sample(fragments, 2)

            result = _run_with_watchdog(
                _one_attempt, mol1, mol2, min_atoms, max_atoms, generated_smiles
            )
            if result is None:
                continue
            smi, reparsed = result

            # All validation (conjugation check + round-trip) already happened
            # inside _one_attempt(), under the watchdog. Just record + stream it.
            generated_smiles.add(smi)
            out_f.write(smi + "\n")
            if draw_sample > 0:
                recent_mols.append(reparsed)

            if len(generated_smiles) % progress_every == 0:
                out_f.flush()
                elapsed = time.time() - t0
                print(
                    f"Generated: {len(generated_smiles)} / {target} "
                    f"(attempts: {attempts}, elapsed: {elapsed:.0f}s)",
                    flush=True,
                )

            if attempts % 20000 == 0:
                elapsed = time.time() - t0
                print(
                    f"...progress check: attempts={attempts}, unique so far={len(generated_smiles)}, "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    return generated_smiles, list(recent_mols), attempts


# --------------------------------------------------------------------------
# Image rendering (sanity-check sample, not all 10k)
# --------------------------------------------------------------------------
def draw_sample(molecules, out_dir, n_images):
    if n_images <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    sample = molecules[-n_images:] if len(molecules) >= n_images else molecules
    for i, mol in enumerate(sample):
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        with open(os.path.join(out_dir, f"mol_{i}.png"), "wb") as f:
            f.write(drawer.GetDrawingText())
    print(f"Saved {len(sample)} sample images to: {out_dir}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate fully conjugated PAH/GQD structures by fusing input fragments."
    )
    p.add_argument(
        "--input",
        default="two_fused_fragments.smi",
        help="Path to input .smi file with fragment SMILES (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write results into (default: %(default)s)",
    )
    p.add_argument("--target", type=int, default=10000, help="Number of unique structures to generate")
    p.add_argument(
        "--max-attempts",
        type=int,
        default=2_000_000,
        help="Hard cap on fusion attempts (the strict conjugation filter rejects "
        "most attempts, so this needs to be large; default: %(default)s)",
    )
    p.add_argument("--min-atoms", type=int, default=10, help="Minimum heavy-atom count to keep")
    p.add_argument("--max-atoms", type=int, default=120, help="Maximum heavy-atom count to keep")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument(
        "--draw-sample",
        type=int,
        default=50,
        help="Number of generated structures to render as PNGs for a quick visual sanity check "
        "(0 to skip; default: %(default)s)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    fragments = load_fragments(args.input)
    print(f"Loaded {len(fragments)} fragments from {args.input}")
    if len(fragments) < 2:
        print("ERROR: need at least 2 valid fragments to fuse.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    out_smi = os.path.join(args.output_dir, "generated_gqds.smi")

    print(
        f"Generating up to {args.target} unique, fully conjugated structures "
        f"(max_attempts={args.max_attempts})..."
    )
    generated_smiles, recent_molecules, attempts = generate(
        fragments,
        target=args.target,
        max_attempts=args.max_attempts,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        out_smi_path=out_smi,
        seed=args.seed,
        draw_sample=args.draw_sample,
    )

    print("\n✅ FINAL RESULT")
    print("Total unique, fully conjugated SMILES:", len(generated_smiles))
    print("Total attempts:", attempts)
    if len(generated_smiles) < args.target:
        print(
            "NOTE: target not reached within max_attempts. This fragment set may not "
            "have enough distinct valid fusion sites to reach the target while staying "
            "fully conjugated. Try increasing --max-attempts, or adding more/varied "
            "input fragments."
        )
    print("Saved SMILES to:", out_smi, "(written incrementally during the run)")

    # Optional: quick visual sanity-check sample (only the most recent
    # `draw_sample` structures -- kept in a bounded buffer during the run
    # so memory doesn't grow with the target size).
    img_dir = os.path.join(args.output_dir, "sample_images")
    draw_sample(recent_molecules, img_dir, args.draw_sample)


if __name__ == "__main__":
    main()
