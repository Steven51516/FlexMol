#name : 7_CNN_ESM_GCN_Cat.py
import sys
import os
sys.path.append("/root/FlexMol")

from FlexMol.dataset.loader import *
from FlexMol.encoder.FM import *
from FlexMol.task import *


device = 'cuda:6'
epoch = 100
patience = 10 
lr = 0.0001
batch_size=64
dir = "data/DAVIS/"
name = dir.split('/')[-2]
#output the device, epoch, patience, lr, batch_size, dir imformation
print("device:", device, "epoch:", epoch, "patience:", patience, "lr:", lr, "batch_size:", batch_size, "data_dir:", dir)

train = load_DAVIS(dir+"train.txt")
val = load_DAVIS(dir+"val.txt")
test = load_DAVIS(dir+"test.txt")


num = 5
for i in range(num):
    #set the seed in torch
    torch.manual_seed(i)
    FM = FlexMol()
    de = FM.init_drug_encoder("CNN", output_feats = 128) 
    pe = FM.init_prot_encoder("GCN_ESM", pdb=True, hidden_feats=[128, 128, 128], data_dir = dir + "pdb/", output_feats = 128, pickle_dir = '/root/FlexMol/pickles/davis/esm_graph')
    dp = FM.set_interaction([de, pe], "cat")
    dp = FM.apply_mlp(dp, head=1, hidden_layers = [512, 512, 256])
    FM.build_model()
    trainer = BinaryTrainer(FM, task = "DTI", early_stopping="roc-auc", test_metrics=["roc-auc", "pr-auc"], 
                            device=device, epochs=epoch, patience=patience, lr=lr, batch_size=batch_size)

    if i == 0:
        train, val, test = trainer.prepare_datasets(train_df=train, val_df=val, test_df=test)
    trainer.train(train, val)
    now = i
    #cheak weather the dir is exist
    while(os.path.exists('./result/'+name+ '/7/'+str(now) + '/')):
        now=now+1
    trainer.metrics_dir = './result/'+name+ '/7/'+str(now) + '/'
    trainer.test(test)