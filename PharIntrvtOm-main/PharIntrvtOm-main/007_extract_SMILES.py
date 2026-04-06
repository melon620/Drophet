from rdkit import Chem
import requests

def get_smiles_from_pubchem(name):
    """Get SMILES from PubChem using the compound name."""
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
    url = f"{base_url}{name}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        print(f"Error fetching SMILES for {name}: {response.status_code}")
        return None

def get_canonical_smiles(smiles):
    """Convert SMILES to canonical SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f"Invalid SMILES string: {smiles}")
        return None

def convert_names_to_canonical_smiles(names):
    """Convert a list of compound names to canonical SMILES."""
    canonical_smiles_list = []
    for name in names:
        smiles = get_smiles_from_pubchem(name)
        if smiles:
            canonical_smiles = get_canonical_smiles(smiles)
            canonical_smiles_list.append((name, canonical_smiles))
        else:
            canonical_smiles_list.append((name, None))
    return canonical_smiles_list

# Example usage
compound_names = ["Dabrafenib Mesylate", "Trametinib Dimethyl Sulfoxide"]
canonical_smiles = convert_names_to_canonical_smiles(compound_names)

for name, smiles in canonical_smiles:
    print(f"{name}: {smiles}")
