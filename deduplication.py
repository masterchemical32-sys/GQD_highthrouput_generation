from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

from rdkit import Chem
from rdkit import RDLogger


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def main() -> int:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Remove duplicate/equivalent SMILES from a .smi file.")
    ap.add_argument("--input", default="all_fused_fragments.smi")
    ap.add_argument("--output", default="unique_fragments.smi")
    ap.add_argument("--col", type=int, default=0,
                     help="Which tab-separated column holds the SMILES (0-indexed).")
    args = ap.parse_args()

    seen: Set[str] = set()
    out_lines: List[str] = []
    total = 0
    kept = 0

    for raw in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        total += 1

        parts = line.split("\t")
        if args.col >= len(parts):
            continue
        smi_raw = parts[args.col].strip()

        canon = canonical_smiles(smi_raw)
        key = canon if canon is not None else smi_raw  # fallback: exact string if unparsable

        if key in seen:
            continue
        seen.add(key)
        kept += 1
        out_lines.append(line)

    Path(args.output).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"Read {total} lines, kept {kept} unique fragments, wrote to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())