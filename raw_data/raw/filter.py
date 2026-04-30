import pickle

import pandas as pd
from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover

from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Define a list of allowed atom symbols
# ATOM_SYMBOLS = ["H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
ATOM_SYMBOLS = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]

# Function to standardize a SMILES string
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    
    mol = Chem.RemoveHs(mol)
    # Try to create a molecule from the SMILES string
    if Descriptors.MolWt(mol) > 2000:
        return 0

    # Check if the molecule has allowed atoms only and has more than one atom
    if (
        any(atom.GetSymbol() not in ATOM_SYMBOLS for atom in mol.GetAtoms())
        or mol.GetNumAtoms() == 1
    ):
        return 0

    # Standardize the molecule
    s = Standardizer()
    mol = s.standardize(mol)
    mol = s.fragment_parent(mol)

    # Remove salts from the molecule
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    # Convert the molecule back to a SMILES string
    return Chem.MolToSmiles(mol)

def standardize_smiles_v2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol or mol.GetNumAtoms() <= 1: return 0
    
    # 1. Fragment/Salt Removal & Heavy Atom Check
    mol = rdMolStandardize.FragmentParent(mol)
    if Chem.Descriptors.MolWt(mol) > 2000: return 0

    # 2. Neutralize Charges
    mol = rdMolStandardize.Uncharger().uncharge(mol)

    # 3. Canonical Tautomer (Resolves Keto/Enol)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

    # 4. Handle Stereochemistry
    # Uncomment below to ignore stereo (treats [C@H] and [C] as identical)
    # Chem.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

def smiles_to_inchikey(smiles, standardized=False):
    # Load the molecule from SMILES
    if standardized:
        mol=standardize_smiles_v2(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        # Generate the InChIKey
        return Chem.MolToInchiKey(mol)
    else:
        return 0

if __name__ == "__main__":
    # name = "coconut_complete-10-2024.csv"
    name = "coconut_csv-04-2026.csv"
    df = pd.read_csv(name)
    smiles_list = df['canonical_smiles'].tolist()

    # Standardize all SMILES strings in the list
    with Pool(cpu_count()*2) as pool:
        clean_smiles = list(tqdm(pool.map(standardize_smiles, smiles_list), total=len(smiles_list)))

    # Remove duplicates and 0 values
    clean_smiles = list(set(clean_smiles) - {0})

    # Print the length of the cleaned SMILES list and check if it contains 0
    print(len(clean_smiles))  # 405468
    print(0 in clean_smiles)  # False

    # Save the cleaned SMILES list to a pickle file
    #plk = "pretrain_smiles.pkl"
    plk = "2026_pretrain_smiles.pkl"
    with open(plk, "wb") as f:
        pickle.dump(clean_smiles, f, protocol=4)