──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Generate Morgan fingerprints using RDKit
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

df = pd.read_csv('Clozapine_dataset.csv')

# Use the new MorganGenerator (fixes deprecation warning)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Function to convert SMILES to fingerprint
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = morgan_gen.GetFingerprintAsNumPy(mol)
    return np.array(fp)

# Generate fingerprints
fp_drug     = np.vstack(df['SMILES_0'].apply(smiles_to_fingerprint))
fp_solvent1 = np.vstack(df['SMILES_1'].apply(smiles_to_fingerprint))
fp_solvent2 = np.vstack(df['SMILES_2'].apply(smiles_to_fingerprint))



──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Generate fingerprints using Mordred descriptors
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
import pandas as pd
from rdkit import Chem
import numpy as np
from mordred import Calculator, descriptors
from mordred import (
    SLogP, TopoPSA, HydrogenBond, McGowanVolume,
    Polarizability, Weight, LogS, Lipinski,
    MoeType, CPSA, RotatableBond, RingCount,
    Aromatic, KappaShapeIndex, Constitutional, EState,
    BCUT, VdwVolumeABC, TopologicalIndex, ZagrebIndex,
    ABCIndex, AcidBase, BondCount, CarbonTypes, Chi, 
    EccentricConnectivityIndex, TopologicalCharge
)

# Only calculate specific descriptor categories(but you can add other descriptors that you need)
calc = Calculator([
    SLogP, TopoPSA, HydrogenBond, McGowanVolume,
    Polarizability, Weight, LogS, Lipinski,
    MoeType, CPSA, RotatableBond, RingCount,
    Aromatic, KappaShapeIndex, Constitutional, EState,
    BCUT, VdwVolumeABC, TopologicalIndex, ZagrebIndex,
    ABCIndex, AcidBase, BondCount, CarbonTypes, Chi, 
    EccentricConnectivityIndex, TopologicalCharge
], ignore_3D=True)

df = pd.read_csv('Clozapine_dataset.csv')

# Function to convert SMILES to mordred descriptors
def smiles_to_mordred(smiles):
    if pd.isna(smiles) or smiles == '':
        return np.zeros(len(calc.descriptors))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(calc.descriptors))
    result = calc(mol)
    # Convert to numeric, replace errors with 0
    values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in result]
    return np.array(values)


# Generate mordred descriptors
fp_solvent1 = np.vstack(df['SMILES_1'].apply(smiles_to_mordred))
fp_solvent2 = np.vstack(df['SMILES_2'].apply(smiles_to_mordred))


──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Generate fingerprints using ChemBERTa
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv('Clozapine_dataset.csv')

# ── Load ChemBERTa model and tokenizer ────────────────────────────────────────
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer   = AutoTokenizer.from_pretrained(model_name)
model       = AutoModel.from_pretrained(model_name)
model.eval()

# ── Use GPU if available ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Using device: {device}")

# ── Function to convert SMILES to ChemBERTa embedding ─────────────────────────
def smiles_to_chemberta(smiles, batch_size=32):
    """
    Convert a list/series of SMILES to ChemBERTa embeddings.
    Returns a numpy array of shape (n_molecules, 768).
    """
    embeddings = []
    smiles_list = list(smiles)

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]

        # Replace NaN or empty with a dummy SMILES
        batch = [s if isinstance(s, str) and s != '' else 'C' for s in batch]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            output = model(**tokens)

        # Use [CLS] token embedding as the molecular fingerprint
        cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

        if i % 100 == 0:
            print(f"  Processed {i}/{len(smiles_list)} molecules...")

    return np.vstack(embeddings)

# ── Generate ChemBERTa embeddings ─────────────────────────────────────────────
print("Generating embeddings for solvent 1...")
fp_solvent1 = smiles_to_chemberta(df['SMILES_1'])  # shape (684, 768)

print("Generating embeddings for solvent 2...")
fp_solvent2 = smiles_to_chemberta(df['SMILES_2'])  # shape (684, 768)

print(f"fp_solvent1 shape: {fp_solvent1.shape}")
print(f"fp_solvent2 shape: {fp_solvent2.shape}")
