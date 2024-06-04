
import dgl
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
import logging
class SuppressWarnings(object):
    def __enter__(self):
        self._original_level = logging.root.level
        logging.root.setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.root.setLevel(self._original_level)
        
with SuppressWarnings():
    import deepchem
import numpy as np
import os
import pandas as pd

pk = deepchem.dock.ConvexHullPocketFinder()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 6, 5, 6, 1) --> total 28


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H


def process_pocket_single(pdb_file, data_dir, max_pockets = 30):
    pdb_file = os.path.join(data_dir, f"{pdb_file}.pdb")
    m = Chem.MolFromPDBFile(pdb_file)
    if m is None:
        m = Chem.MolFromPDBFile(pdb_file, sanitize = False)
    am = GetAdjacencyMatrix(m)

    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        graph = dgl.DGLGraph(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)
        if(len(constructed_graphs) > max_pockets):
            break

    return constructed_graphs



def process_deepchem_pocket(pdb_file, data_dir, max_pockets = 30):
    pdb_file_dir = os.path.join(data_dir, pdb_file)
    constructed_graphs = []
    if os.path.isdir(pdb_file_dir):
        for file_name in os.listdir(pdb_file_dir):
            if file_name.endswith(".pdb"):
                pdb_id = file_name[:-4]
                graphs = process_pocket_single(pdb_id, pdb_file_dir, max_pockets)
                constructed_graphs.extend(graphs)
                if(len(constructed_graphs) > max_pockets):
                   break
    else:
        constructed_graphs = process_pocket_single(pdb_file, data_dir, max_pockets)
    
    while len(constructed_graphs) < max_pockets:
        virtual_graph = dgl.graph(([], []))
        virtual_graph.add_nodes(1)
        virtual_graph.ndata['h'] = torch.zeros((1, 30))
        constructed_graphs.append(virtual_graph)

    constructed_graphs = dgl.batch(constructed_graphs[:max_pockets])

    return constructed_graphs


