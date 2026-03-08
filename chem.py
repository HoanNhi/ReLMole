
import numpy as np
# from scipy.sparse import csr_matrix
# import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict, deque
from copy import deepcopy
from numpy import random
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw


# feature dim
ATOM_DIM = 101
BOND_DIM = 11
FG_DIM = 73
FG_EDGE_DIM = ATOM_DIM

ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}

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

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
            'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()] +
        [atom.IsInRing()]
    )


def get_bond_feature(bond):
    return np.array(
        one_of_k_encoding(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
        [bond.GetIsConjugated()] +
        one_of_k_encoding(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
    )


def get_fg_feature(fg_prop):
    return np.array(
        one_of_k_encoding_unk(fg_prop['#C'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#O'], range(6)) +  # 0-5, 5+
        one_of_k_encoding_unk(fg_prop['#N'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#P'], range(6)) +
        one_of_k_encoding_unk(fg_prop['#S'], range(6)) +
        [fg_prop['#X'] > 0] +
        [fg_prop['#UNK'] > 0] +
        one_of_k_encoding_unk(fg_prop['#SINGLE'], range(11)) +  # 0-10, 10+
        one_of_k_encoding_unk(fg_prop['#DOUBLE'], range(8)) +  # 0-6, 6+
        one_of_k_encoding_unk(fg_prop['#TRIPLE'], range(8)) +
        one_of_k_encoding_unk(fg_prop['#AROMATIC'], range(8)) +
        [fg_prop['IsRing']]
    )

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

def mol_to_graphs(mol):
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
            new_mark_i, _= remove_peptide_nitrogen(marks[i], mol)
            marks[i] = set(new_mark_i)

    flattened_marks = set([i for atoms in marks for i in atoms])
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx

    for atom in flattened_marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs)-1]

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
        elif not (a1 in flattened_marks or a2 in flattened_marks):  # both atoms are unmarked, i.e. a trivial C-C single bond
            # add single bond to fgs
            if len(atom2fg[a1]) == 0:
                fgs.append({a1})
                atom2fg[a1].append(len(fgs)-1)
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
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]
    for i in range(len(fgs)): # update atom2fg
        for atom in fgs[i]:
            atom2fg[atom].append(i)

    # <editor-fold desc="generate atom-level graph and get FG's properties">
    atom_features, bond_list, bond_features = [], [], []
    fg_prop = [defaultdict(int) for _ in range(len(fgs))]  # prop: atom: #C, #O, #N, #P, #S, #X, #UNK; bond: #SINGLE, #DOUBLE, #TRIPLE, #AROMATIC, IsRing
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature(atom).tolist())
        elem = atom.GetSymbol()
        if elem in ['C', 'O', 'N', 'P', 'S']:
            key = '#'+elem
        elif elem in ['F', 'Cl', 'Br', 'I']:
            key = '#X'
        else:
            key = '#UNK'
        for fg_idx in atom2fg[atom.GetIdx()]:
            fg_prop[fg_idx][key] += 1
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_list.extend([[a1, a2], [a2, a1]])
        bond_features.extend([get_bond_feature(bond).tolist()] * 2)
        key = '#'+str(bond.GetBondType())
        for fg_idx in (set(atom2fg[a1]) & set(atom2fg[a2])):
            fg_prop[fg_idx][key] += 1
            if bond.IsInRing():
                fg_prop[fg_idx]['IsRing'] = 1
    # </editor-fold>

    # <editor-fold desc="generate FG-level graph">
    fg_features, fg_edge_list, fg_edge_features = [], [], []
    for i in range(len(fgs)):
        fg_features.append(get_fg_feature(fg_prop[i]).tolist())
        for j in range(i+1, len(fgs)):
            shared_atoms = list(fgs[i] & fgs[j])
            if len(shared_atoms) > 0:
                fg_edge_list.extend([[i, j], [j, i]])
                if len(shared_atoms) == 1:
                    fg_edge_features.extend([atom_features[shared_atoms[0]]] * 2)
                else:  # two rings shared 2 atoms, i.e. 1 edge
                    assert len(shared_atoms) == 2
                    ef = [(i+j)/2 for i, j in zip(atom_features[shared_atoms[0]], atom_features[shared_atoms[1]])]
                    fg_edge_features.extend([ef] * 2)
    # </editor-fold>

    atom2fg_list = []
    for fg_idx in range(len(fgs)):
        for atom_idx in fgs[fg_idx]:
            atom2fg_list.append([atom_idx, fg_idx])

    atomHighLights = {}
    atomRadii = {}
    for fg in fgs:
        color = tuple(random.rand() for _ in range(3))
        for a in fg:
            atomHighLights[a] = [color] if a in flattened_marks else [(0,0,0)]
            atomRadii[a] = 0.5

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.DrawMoleculeWithHighlights(
        mol,
        "",
        atomHighLights,
        {},
        atomRadii,
        {}
    )

    drawer.FinishDrawing()

    with open("highlight.png", "wb") as f:
        f.write(drawer.GetDrawingText())
    return atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list


if __name__ == '__main__':
    smiles = "CC[C@H](C)[C@@H]1NC(=O)[C@H](C(C)C)N(C)C(=O)[C@@H]2C[C@@]3(O)C4=CC=CC=C4N[C@H]3N2C(=O)CN(C)C(=O)[C@H]([C@@H](C)CC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@@H](O)N(C)C(=O)[C@H]([C@@H](C)[C@H](C)OC(=O)CC(C)(C)O)N(C)C(=O)C(C(C)(C)O)N(C)C(=O)CN(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](C(C)C)N(C)C1=O"
    mol = Chem.MolFromSmiles(smiles)
    atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
    pass