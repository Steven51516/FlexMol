from sklearn.preprocessing import OneHotEncoder
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
from itertools import zip_longest
import re

from FlexMol.util.biochem.protein.pockets import *
from FlexMol.util.biochem.protein.gvp import *
from FlexMol.util.biochem.BPEEncoder import BPEEncoder
from FlexMol.util.biochem.pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, \
CalculateConjointTriad, GetQuasiSequenceOrder, CalculateCTD, CalculateAutoTotal

from .base import Featurizer

# several methods adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

class ProteinOneHotFeaturizer(Featurizer):
    def __init__(self, max_seq = 1000):
        super(ProteinOneHotFeaturizer, self).__init__()
        amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                      'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
        self.onehot_enc = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
        self.amino_char = amino_char
        self.transform_modes["initial"] = self.initial_transform
        self.transform_modes["loadtime"] = self.loadtime_transform
        self.max_seq = max_seq

    def initial_transform(self, x):
        temp = list(x.upper())
        temp = [i if i in self.amino_char else '?' for i in temp]
        if len(temp) <  self.max_seq:
            temp = temp + ['?'] * ( self.max_seq - len(temp))
        else:
            temp = temp[: self.max_seq]
        return temp

    def loadtime_transform(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x = self.initial_transform(x)
        return self.loadtime_transform(x)



class ProteinESPFTransFeaturizer(Featurizer):
    def __init__(self, vocab_path="FlexMol/util/biochem/ESPF/protein_codes_uniprot_2000.txt",
                 subword_map_path="FlexMol/util/biochem/ESPF/subword_units_map_uniprot_2000.csv", max_d=545):
        super(ProteinESPFTransFeaturizer, self).__init__()
        self.prot_encoder = BPEEncoder(vocab_path, subword_map_path)
        self.max_d = max_d

    def transform(self, x):
        return self.prot_encoder.encode(x, self.max_d)


class ProteinESPFFeaturizer(Featurizer):
    def __init__(self, vocab_path="FlexMol/util/biochem/ESPF/protein_codes_uniprot_2000.txt",
                 subword_map_path="FlexMol/util/biochem/ESPF/subword_units_map_uniprot_2000.csv"):
        super(ProteinESPFFeaturizer, self).__init__()
        self.prot_encoder = BPEEncoder(vocab_path, subword_map_path)

    def transform(self, x):
        return self.prot_encoder.encode(x, mask = False)



class ProteinCTDFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = CalculateCTD(x)
        except:
            print('CTD fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((147, ))
        return np.array(features)



class ProteinAutoCorrFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features =  CalculateAutoTotal(x)
        except:
            print('Auto Correlation fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((720, ))
        return np.array(features)



class ProteinAACFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = CalculateAADipeptideComposition(x)
        except:
            print('AAC fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((8420, ))
        return np.array(features)


class ProteinPAACFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = _GetPseudoAAC(x)
        except:
            print('PesudoAAC fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((30, ))
        return np.array(features)



class ProteinQuasiFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = GetQuasiSequenceOrder(x)
        except:
            print('Quasi-seq fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((100, ))
        return np.array(features)


class ProteinCTFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = CalculateConjointTriad(x)
        except:
            print('Conjoint Triad fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((343, ))
        return np.array(features)



class ProteinGraphFeaturizer(Featurizer):
    def __init__(self, data_dir):
        super(ProteinGraphFeaturizer, self).__init__()
        from FlexMol.util.biochem.protein.prot_graph import create_prot_dgl_graph
        self.data_dir = data_dir
        self._transform = create_prot_dgl_graph

    def transform(self, x):
        return self._transform(x, self.data_dir)
    

class ProteinGraphESMFeaturizer(Featurizer):
    def __init__(self, data_dir):
        super(ProteinGraphESMFeaturizer, self).__init__()
        from FlexMol.util.biochem.protein.prot_graph import create_prot_esm_dgl_graph
        self.data_dir = data_dir
        self._transform = create_prot_esm_dgl_graph

    def transform(self, x):
        return self._transform(x, self.data_dir)



class PocketDeepChemFeaturizer(Featurizer):
    def __init__(self, data_dir, pocket_num = 30):
        super(PocketDeepChemFeaturizer, self).__init__()
        self.max_pockets = pocket_num
        self.data_dir = data_dir
    def transform(self, x):
        return process_deepchem_pocket(x, self.data_dir, self.max_pockets)



class ProteinGVPFeaturizer(Featurizer):
    def __init__(self, data_dir, pocket_num = 30):
        super(ProteinGVPFeaturizer, self).__init__()
        self.max_pockets = pocket_num
        self.data_dir = data_dir
    def transform(self, x):
        return create_gvp_graph(x, self.data_dir)

        

class ProteinTorchDrug(Featurizer):
    def __init__(self, data_dir):
        super(ProteinTorchDrug, self).__init__()
        self.data_dir = data_dir
    def transform(self, x):   
        from torchdrug.data.protein import Protein
        return Protein.from_pdb(self.data_dir + x + ".pdb")



#TODO add batch transform mathod
class ProteinESMFeaturizer(Featurizer):
    def __init__(self):
        super(ProteinGraphFeaturizer, self).__init__()
        from FlexMol.util.biochem.protein.prot_graph import create_prot_dgl_graph
        self._transform = create_prot_dgl_graph
    def transform(self, x):   
        embeddings =  self._transform(x)
        mean_embedding = embeddings[0][1:-1].mean(axis=0)
        return mean_embedding


#TODO support custom precision
class BaseProtTransFeaturizer(Featurizer):
    def __init__(self, device: str = 'cpu'):
        super(BaseProtTransFeaturizer, self).__init__()
        self.device = torch.device(device)
        self.model = self.load_model().to(self.device)
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        raise NotImplementedError("load_model method needs to be implemented by the child class!")

    def load_tokenizer(self):
        raise NotImplementedError("load_tokenizer method needs to be implemented by the child class!")

    def transform(self, sequence: str) -> np.ndarray:
        embeddings = self._embed_batch_impl([sequence])
        return embeddings[0].mean(axis=0)

    def transform_batch(self, x, mode):
        unique_values = np.unique(x)
        embeddings = self._embed_batch_impl(unique_values)
        unique_transformed = [embedding.mean(axis=0) for embedding in embeddings]
        mapping = dict(zip(unique_values, unique_transformed))
        return [mapping[item] for item in x]

    def _embed_batch_impl(self, batch):
        seq_lens = [len(seq) for seq in batch]
        batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch]
        batch = [" ".join(list(seq)) for seq in batch]

        ids = self.tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        with torch.no_grad():
            embeddings = self.model(input_ids=tokenized_sequences, attention_mask=attention_mask)

        embeddings = embeddings.last_hidden_state.cpu().numpy()

        results = []
        for seq_num, seq_len in zip_longest(range(len(embeddings)), seq_lens):
            embedding = embeddings[seq_num][1:seq_len + 1]
            assert seq_len == embedding.shape[0], f"Sequence length mismatch: {seq_len} vs {embedding.shape[0]}"
            results.append(embedding)

        return results


class ProteinT5Featurizer(BaseProtTransFeaturizer):
    def __init__(self, device: str = 'cpu'):
        super(ProteinT5Featurizer, self).__init__(device=device)

    def load_model(self):
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        if self.device == torch.device("cpu"):
            model.to(torch.float32)
        return model

    def load_tokenizer(self):
        return T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, legacy=True)


class ProteinBertFeaturizer(BaseProtTransFeaturizer):
    def __init__(self, device: str = 'cpu'):
        super(ProteinBertFeaturizer, self).__init__(device=device)

    def load_model(self):
        return BertModel.from_pretrained("Rostlab/prot_bert_bfd")

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)


class ProteinAlbertFeaturizer(BaseProtTransFeaturizer):
    def __init__(self, device: str = 'cpu'):
        super(ProteinAlbertFeaturizer, self).__init__(device=device)

    def load_model(self):
        return AlbertModel.from_pretrained("Rostlab/prot_albert")

    def load_tokenizer(self):
        return AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)