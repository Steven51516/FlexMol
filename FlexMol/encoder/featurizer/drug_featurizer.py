
from FlexMol.util.biochem.BPEEncoder import BPEEncoder
from .base import Featurizer
from sklearn.preprocessing import OneHotEncoder
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdReducedGraphs
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_complete_graph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
import torch
from FlexMol.util.biochem.pybiomed_helper import calcPubChemFingerAll
import torch.nn as nn


# several methods adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

class DrugOneHotFeaturizer(Featurizer):
    def __init__(self, max_seq = 100):
        super(DrugOneHotFeaturizer, self).__init__()
        smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
        self.onehot_enc = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
        self.smiles_char = smiles_char
        self.transform_modes["initial"] = self.initial_transform
        self.transform_modes["loadtime"] = self.loadtime_transform
        self.max_seq = 100

    def initial_transform(self, x):
        temp = list(x)
        temp = [i if i in self.smiles_char else '?' for i in temp]
        if len(temp) <  self.max_seq:
            temp = temp + ['?'] * (self.max_seq - len(temp))
        else:
            temp = temp[:self.max_seq]
        return temp

    def loadtime_transform(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x = self.initial_transform(x)
        return self.loadtime_transform(x)



class BaseGraphFeaturizer(Featurizer):
    def __init__(self, virtual_nodes=False, max_node=50):
        super().__init__()
        self.virtual_nodes = virtual_nodes
        self.max_node = max_node

    def add_virtual_nodes(self, graph):
        actual_node_feats = graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_node - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        graph.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, actual_node_feats.shape[1] - 1), torch.ones(num_virtual_nodes, 1)), 1)
        graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        graph = graph.add_self_loop()
        return graph



class DrugCanonicalFeaturizer(BaseGraphFeaturizer):
    def __init__(self, virtual_nodes=False, max_node=50):
        super().__init__(virtual_nodes, max_node)
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.transform_func = partial(smiles_to_bigraph,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=True)

    def transform(self, x):
        graph = self.transform_func(x)
        if self.virtual_nodes:
            graph = self.add_virtual_nodes(graph)
        return graph


class DrugAttentiveFPFeaturizer(BaseGraphFeaturizer):
    def __init__(self, virtual_nodes=False, max_node=50):
        super().__init__(virtual_nodes, max_node)
        self.node_featurizer = AttentiveFPAtomFeaturizer()
        self.edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
        self.transform_func = partial(smiles_to_bigraph,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=True)

    def transform(self, x):
        graph = self.transform_func(x)
        if self.virtual_nodes:
            graph = self.add_virtual_nodes(graph)
        return graph


class Drug3dFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
        from dgllife.data.alchemy import alchemy_nodes, alchemy_edges
        self.node_featurizer = alchemy_nodes
        self.edge_featurizer = alchemy_edges
        self.transform_func = partial(mol_to_complete_graph,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer)

    def generate_single_conformer(self, smiles):
        """Generates a molecule with a single conformer from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        conformers = mol.GetConformers()
        assert len(conformers) == 1, "More than one conformer generated"
        return mol

    def transform(self, x):
        mol = self.generate_single_conformer(x)
        graph = self.transform_func(mol)
        return graph


class MorganFeaturizer(Featurizer):
    def __init__(self, radius=2, nbits=1024):
        super().__init__()
        self.radius = radius
        self.nBits = nbits

    def transform(self, s):
        try:
            mol = Chem.MolFromSmiles(s)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except:
            print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
            features = np.zeros((self.nBits,))
        return features


class DrugESPFTransFeaturizer(Featurizer):
    def __init__(self, vocab_path="FlexMol/util/biochem/ESPF/drug_codes_chembl_freq_1500.txt",
                 subword_map_path="FlexMol/util/biochem/ESPF/subword_units_map_chembl_freq_1500.csv",
                 max_d=50):
        super(DrugESPFTransFeaturizer, self).__init__()
        self.drug_encoder = BPEEncoder(vocab_path, subword_map_path)
        self.max_d = max_d

    def transform(self, x):
        return self.drug_encoder.encode(x, self.max_d)



class DrugESPFFeaturizer(Featurizer):
    def __init__(self, vocab_path="FlexMol/util/biochem/ESPF/drug_codes_chembl_freq_1500.txt",
                 subword_map_path="FlexMol/util/biochem/ESPF/subword_units_map_chembl_freq_1500.csv"):
        super(DrugESPFFeaturizer, self).__init__()
        self.drug_encoder = BPEEncoder(vocab_path, subword_map_path)

    def transform(self, x):
        return self.drug_encoder.encode(x, mask = False)


class ErGFeaturizer(Featurizer):
    """Featurizer for Extended-reduced Graph (ErG) Fingerprints."""
    def __init__(self):
        super().__init__()

    def transform(self, s):
        try:
            mol = Chem.MolFromSmiles(s)
            features = np.array(rdReducedGraphs.GetErGFingerprint(mol))
        except:
            print('RDKit cannot find this SMILES for ErG: ' + s + ' convert to all 0 features')
            features = np.zeros((315,))
        return features
    

class DaylightFeaturizer(Featurizer):
    """Featurizer for Daylight-like Fingerprints using RDKit's built-in method."""
    def __init__(self, num_finger=2048):
        super().__init__()
        self.num_finger = num_finger

    def transform(self, s):
        try:
            mol = Chem.MolFromSmiles(s)
            bv = FingerprintMols.FingerprintMol(mol)
            temp = tuple(bv.GetOnBits())
            features = np.zeros((self.num_finger,))
            features[np.array(temp)] = 1
        except:
            print('RDKit not found this SMILES: ' + s + ' convert to all 0 features')
            features = np.zeros((self.num_finger,))
        return np.array(features)
    


class PubChemFeaturizer(Featurizer):
    """Featurizer for PubChem Fingerprints."""

    def transform(self, x):
        try:
            features = calcPubChemFingerAll(x)
        except:
            print('pubchem fingerprint not working for smiles: ' + x + ' convert to 0 vectors')
            features = np.zeros((881, ))
        return np.array(features)



class ChemBERTaFeaturizer(Featurizer):
    """Featurizer using ChemBERTa model to generate molecular embeddings."""
    def __init__(self, model_path="DeepChem/ChemBERTa-77M-MTR", mode='mean'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model._modules["lm_head"] = nn.Identity() 
        self.mode = mode

    def transform(self, smiles):
        """Transforms a SMILES string into a ChemBERTa embedding."""
        try:
            encoded_input = self.tokenizer(smiles, return_tensors="pt")
            model_output = self.model(**encoded_input)
            if self.mode == 'cls':
                # Use the embedding from the CLS token
                embedding = model_output[0][:, 0, :]  # CLS token is the first token
            elif self.mode == 'mean':
                # Calculate the mean of all token embeddings
                embedding = torch.mean(model_output[0], dim=1)
            else:
                raise ValueError("Unsupported mode. Choose 'cls' or 'mean'.")
            return embedding.squeeze().tolist()
        except Exception as e:
            print(f'Error processing SMILES {smiles}: {str(e)}')
            return []


