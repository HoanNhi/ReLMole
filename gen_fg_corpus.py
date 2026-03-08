import os, json, time

from rdkit.Chem import rdmolops
from tqdm import tqdm
from collections import defaultdict, deque
from rdkit import Chem

PATT = {
    'HETEROATOM': '[!#6]',
    'DOUBLE_TRIPLE_BOND': '*=,#*',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]'
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}

STD_AMINO = {
    "amino_alanine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH3X4])[CX3](=[OX1])[O,N]",
    "amino_arginine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3])[CX3](=[OX1])[O,N]",
    "amino_asparagine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[NX3H2])[CX3](=[OX1])[O,N]",
    "amino_aspartate": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[O,N]",
    "amino_cysteine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][SX2H,SX1H0-])[CX3](=[OX1])[O,N]",
    "amino_glutamate": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[O,N]",
    "amino_histidine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1)[CX3](=[OX1])[O,N]",
    "amino_isoleucine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH2X4][CH3X4])[CX3](=[OX1])[O,N]",
    "amino_leucine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[O,N]",
    "amino_lysine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0])[CX3](=[OX1])[O,N]",
    "amino_methionine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][SX2][CH3X4])[CX3](=[OX1])[O,N]",
    "amino_phenylalanine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1)[CX3](=[OX1])[O,N]",
    "amino_serine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][OX2H])[CX3](=[OX1])[O,N]",
    "amino_threonine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[OX2H])[CX3](=[OX1])[O,N]",
    "amino_tryptophan": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12)[CX3](=[OX1])[O,N]",
    "amino_tyrosine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1)[CX3](=[OX1])[O,N]",
    "amino_valine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[O,N]",
    "glycine": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[O,N]",
    "proline": "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]"
}
ALL_AMINO = {
    "all_amino": "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]"
}
ALL_AMINO = {k: Chem.MolFromSmarts(v) for k, v in ALL_AMINO.items()}
STD_AMINO = {k: Chem.MolFromSmarts(v) for k, v in STD_AMINO.items()}

def remove_peptide_nitrogen(aa1_atoms, mol):
    # submol_1 = Chem.PathToSubmol(mol, aa1_atoms)
    atom_map = {}
    submol_1 = rdmolops.PathToSubmol(
    mol,
    [bond.GetIdx() for bond in mol.GetBonds()
     if bond.GetBeginAtomIdx() in aa1_atoms and bond.GetEndAtomIdx() in aa1_atoms],
    atomMap=atom_map
    )
    # submol_2 = Chem.PathToSubmol(mol, aa2_atoms)

    atom_map = {v:k for k, v in atom_map.items()}
    # aa1_atoms = list(aa1_atoms)
    # aa2_atoms = list(aa2_atoms)

    peptide_bond = Chem.MolFromSmarts("[C:1](=O)[N:2]")
    substructs = submol_1.GetSubstructMatch(peptide_bond)
    if len(substructs) > 0:
        n_index = substructs[2]
        aa1_atoms.remove(atom_map[n_index])
        return aa1_atoms, peptide_bond
    # elif len(submol_2.GetSubstructMatches(peptide_bond)) > 0:
    #     aa2_atoms = aa2_atoms[:-1]
    #     return aa1_atoms, aa2_atoms, peptide_bond


    peptide_bond = Chem.MolFromSmarts("C(O)N")
    if len(submol_1.GetSubstructMatches(peptide_bond)) > 0:
        aa1_atoms = aa1_atoms[:-1]
        return aa1_atoms, peptide_bond
    # elif len(submol_2.GetSubstructMatches(peptide_bond)) > 0:
    #     aa2_atoms = aa2_atoms[:-1]
    #     return aa1_atoms, aa2_atoms, peptide_bond

    peptide_bond = Chem.MolFromSmarts("C(=C)N")
    carbonxylic = Chem.MolFromSmarts("C(=O)O")
    if (len(submol_1.GetSubstructMatches(peptide_bond)) > 0
            and submol_1.GetSubstructMatches(carbonxylic) == 0):
        aa1_atoms = aa1_atoms[:-1]
        return aa1_atoms, peptide_bond
    # elif len(submol_2.GetSubstructMatches(peptide_bond)) > 0\
    #         and submol_2.GetSubstructMatches(carbonxylic) == 0:
    #     aa2_atoms = aa2_atoms[:-1]
    #     return aa1_atoms, aa2_atoms, peptide_bond

    peptide_bond = Chem.MolFromSmarts("C(C)N")
    carbonxylic = Chem.MolFromSmarts("C(=O)O")
    if (len(submol_1.GetSubstructMatches(peptide_bond)) > 0
            and submol_1.GetSubstructMatches(carbonxylic) == 0):
        aa1_atoms = aa1_atoms[:-1]
        return aa1_atoms, peptide_bond
    # elif len(submol_2.GetSubstructMatches(peptide_bond)) > 0 \
    #         and submol_2.GetSubstructMatches(carbonxylic) == 0:
    #     aa2_atoms = aa2_atoms[:-1]
    #     return aa1_atoms, aa2_atoms, peptide_bond

    return aa1_atoms, peptide_bond

def get_side_chain_atoms(mol, match):
    backbone = set(match)  # backbone atoms
    sidechain_root = match[2]

    visited = set()
    queue = deque([sidechain_root])
    side_atoms = set()

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)

        atomLocal = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atomLocal.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in backbone:
                side_atoms.add(nbr_idx)
                queue.append(nbr_idx)
            # if nbr_idx not in visited:
            #     queue.append(nbr_idx)

    return side_atoms

def get_fg_set(mol):
    """
    Identify FGs and convert to SMILES
    Args:
        mol:
    Returns: a set of FG's SMILES
    """
    fgs = []  # Function Groups

    # <editor-fold desc="identify functional atoms and merge connected ones">
    marks = []
    for patt in ALL_AMINO.values():  # mark functional atoms
        for sub in mol.GetSubstructMatches(patt):
            side_atoms = get_side_chain_atoms(mol, sub)
            side_atoms.update(sub)
            marks.append(side_atoms)

    # peptide_smarts = Chem.MolFromSmarts("C(=O)N")
    # Peptide bond can also be: C(O)N, C(=C)N, C(C)N (the last two need further examination)
    # peptide_matches = set(mol.GetSubstructMatches(peptide_smarts))

    for i in range(len(marks)):
        # for j in range(i+1, len(marks)):
        #     if len(set(marks[i]) & set(marks[j])) == 0:
        #         continue
        new_mark_i, _ = remove_peptide_nitrogen(marks[i], mol)
        marks[i] = set(new_mark_i)

    flattened_marks = set([i for atoms in marks for i in atoms])
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx

    for atom in flattened_marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs) - 1]

    for bond in mol.GetBonds():  # merge FGs
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in flattened_marks and a2 in flattened_marks:
            # a marked atom can belong to one or more FGs. However, the 0th
            # index stores its primary FG.
            assert a1 != a2
            assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
            if any(a1 in mark and a2 in mark for mark in marks):
                # fgs.append({a1, a2})
                fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
                if atom2fg[a1][0] != atom2fg[a2][0]:
                    fgs[atom2fg[a2][0]] = set()
                atom2fg[a2] = [atom2fg[a1][0]]
            # else:
            #     fgs.append({a1, a2})
            #     atom2fg[a1].append(len(fgs) - 1)
            #     atom2fg[a2].append(len(fgs) - 1)

        # elif a1 in flattened_marks:  # only one atom is marked, add neighbour atom to its FG as its environment
        #     assert len(atom2fg[a1]) == 1
        #     # add a2 to a1's FG
        #     fgs[atom2fg[a1][0]].add(a2)
        #     atom2fg[a2].extend(atom2fg[a1])
        # elif a2 in flattened_marks:
        #     # add a1 to a2's FG
        #     assert len(atom2fg[a2]) == 1
        #     fgs[atom2fg[a2][0]].add(a1)
        #     atom2fg[a1].extend(atom2fg[a2])
        elif not (
                a1 in flattened_marks or a2 in flattened_marks):  # both atoms are unmarked, i.e. a trivial C-C single bond
            # add single bond to fgs
            if len(atom2fg[a1]) == 0:
                fgs.append({a1})
                atom2fg[a1].append(len(fgs) - 1)
            fgs[atom2fg[a1][0]].add(a2)
            # if atom2fg[a1][0] != atom2fg[a2][0]:
            #     fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = [atom2fg[a1][0]]
    tmp = []
    for fg in fgs:
        if len(fg) == 0: continue
        # if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing(): continue  # single atom FGs: 1. marked atom only in ring: remove; 2. ion or simple substance: retain
        tmp.append(fg)
    fgs = tmp
    # </editor-fold>

    # fgs.extend(rings)  # final FGs: rings + FGs (not in rings)

    fg_smiles = set()
    for fg in fgs:
        fg_smiles.add(Chem.MolFragmentToSmiles(mol, fg))

    return fg_smiles


if __name__ == '__main__':
    os.chdir('data/ZINC15')

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Splitting mol to FGs...")
    with open('zinc15_250k.txt') as f:
        smiles_list = f.read().splitlines()
    print(f"# mols: {len(smiles_list)}")

    mol2fgs = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fg_smiles = get_fg_set(mol)
        mol2fgs.append(list(fg_smiles))

    with open('mol2fgs_list.json', 'w') as f:
        json.dump(mol2fgs, f)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating FG corpus...")
    with open('mol2fgs_list.json', 'r') as f:
        mol2fgs = json.load(f)
    print(f"# mols: {len(mol2fgs)}")

    fg_dict = defaultdict(int)
    for fgs in tqdm(mol2fgs):
        for fg in fgs:
            fg_dict[fg] += 1
    print(f"# fgs: {len(fg_dict)}")

    fg_corpus = sorted(fg_dict.keys(), key=lambda k: fg_dict[k], reverse=True)
    fg_corpus = fg_corpus[:512]
    print(f"# corpus: {len(fg_corpus)}")
    print(f"freq: {fg_dict[fg_corpus[-1]]}/{len(mol2fgs)}")

    with open('fg_corpus.txt', 'w') as f:
        for fg in fg_corpus:
            f.write(fg+"\n")

