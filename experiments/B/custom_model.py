import sys
sys.path.append("/root/FlexMol")

import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from FlexMol.encoder.enc_layer import *
from FlexMol.encoder.featurizer import DrugCanonicalFeaturizer
from FlexMol.dataset.loader import *
from FlexMol.encoder import FlexMol
from FlexMol.task import BinaryTrainer
from FlexMol.encoder.enc_layer import EncoderLayer

@register_collate_func(dgl.batch)
@register_to_device(True)
class My_GCN(EncoderLayer):
    """
    DGL_GCN is a graph convolutional network implementation using DGL's GCN and WeightedSumAndMax.
    """

    def __init__(self, in_feats=74, hidden_feats=[64, 64, 64], activation=[F.relu, F.relu, F.relu], output_feats=64, device='cpu'):
        super(My_GCN, self).__init__()
        self.device = device
        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation)
        gnn_out_feats = self.gnn.hidden_feats[-1]

        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.output_shape = output_feats
        self.transform = nn.Linear(gnn_out_feats * 2, output_feats)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, bg):
        bg = bg.to(self.device)
        feats = bg.ndata.pop('h')
        feats = feats.to(torch.float32)
        node_feats = self.gnn(bg, feats)
        return self.transform(self.readout(bg, node_feats))




dir = "data/BIOSNAP/"
train_df = load_BIOSNAP(dir+"train.csv").head(2)
val_df = load_BIOSNAP(dir+"val.csv").head(2)
test_df = load_BIOSNAP(dir+"test.csv").head(2)

FM = FlexMol()
FM.register_method("drug", "my_gcn", My_GCN, DrugCanonicalFeaturizer)
de = FM.init_drug_encoder("my_gcn", output_feats = 128)
pocket =  FM.init_prot_encoder("PocketDC", pdb=True, data_dir = dir + "pdb/", num_pockets = 30, output_feats=128, pooling = False, pickle_dir = '/root/FlexMol/pickles/biosnap/pocket')
pe = FM.init_prot_encoder("GCN_ESM", pdb=True, hidden_feats=[128, 128, 128], data_dir = dir + "pdb/", output_feats = 128, pickle_dir = '/root/FlexMol/pickles/biosnap/esm_graph')
att = FM.set_interaction(FM.stack([de, pocket, pe]), "self_attention")
d_final = FM.flatten(FM.select(att, index_start=0))
p_final = FM.flatten(FM.select(att, index_start=31))
output = FM.apply_mlp(FM.cat([d_final, p_final]), hidden_layers=[512, 512, 256], head=1)
FM.build_model()
trainer = BinaryTrainer(
    FM, task = "DTI", early_stopping="roc-auc", test_metrics=["roc-auc", "pr-auc"], 
    device="cuda:0", epochs=25, patience=7, lr=0.0001, batch_size=32, metrics_dir = "result/exp2", checkpoint_dir = "checkpoint/exp2"
)

train, val, test = trainer.prepare_datasets(train_df=train_df, val_df=val_df, test_df=test_df)
trainer.train(train, val)
trainer.test(test)
