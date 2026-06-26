from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

INPUT_FILE = "input_GQDs.smi"      # your 17 SMILES
OUTPUT_FILE = "cleaned_cores_new.smi"


# -------------------------
# KEEP LARGEST AROMATIC CORE
# -------------------------
def keep_largest_aromatic_core(mol):
    try:
        keep_atoms = [a.GetIdx() for a in mol.GetAtoms()
                      if a.GetIsAromatic() and a.GetSymbol() == "C"]

        if not keep_atoms:
            return None

        emol = Chem.RWMol()
        atom_map = {}

        # Add atoms
        for idx in keep_atoms:
            atom_map[idx] = emol.AddAtom(Chem.Atom("C"))

        # Add bonds
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            if a1 in atom_map and a2 in atom_map:
                emol.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

        core = emol.GetMol()
        Chem.SanitizeMol(core)

        # Split into fragments
        frags = Chem.GetMolFrags(core, asMols=True)

        if not frags:
            return None

        # Keep largest fragment
        largest = max(frags, key=lambda m: m.GetNumAtoms())

        # Fix hydrogens
        largest = Chem.AddHs(largest)
        Chem.SanitizeMol(largest)
        largest = Chem.RemoveHs(largest)

        return largest

    except:
        return None


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    cleaned_smiles = []

    with open(INPUT_FILE, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print("Loaded molecules:", len(smiles_list))

    success = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print("Invalid SMILES skipped")
            continue

        core = keep_largest_aromatic_core(mol)

        if core is None:
            print("Core extraction failed")
            continue

        smi_core = Chem.MolToSmiles(core, canonical=True)
        cleaned_smiles.append(smi_core)
        success += 1

    # Remove duplicates
    unique_smiles = sorted(set(cleaned_smiles))

    # Save to .smi
    with open(OUTPUT_FILE, "w") as f:
        for smi in unique_smiles:
            f.write(smi + "\n")

    print("\n✅ DONE")
    print("Successful:", success)
    print("Unique cores:", len(unique_smiles))
    print("Saved to:", OUTPUT_FILE)