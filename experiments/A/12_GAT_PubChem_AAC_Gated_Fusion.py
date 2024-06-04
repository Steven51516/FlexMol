#name : 12_GCN_Morgan_AAC_Gated_Fusion.py
import sys
import os
sys.path.append("/root/FlexMol")

from FlexMol.dataset.loader import *
from FlexMol.encoder.FM import *
from FlexMol.task import *


device = 'cuda:3'
epoch = 100
patience = 10 
lr = 0.0001
batch_size=64
dir = "data/DAVIS/"
name =  dir.split('/')[-2]
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
    de1 = FM.init_drug_encoder("GAT", output_feats = 64)
    de2 = FM.init_drug_encoder("PubChem", output_dim = 64)
    pe = FM.init_prot_encoder("AAC", output_dim=64)
    de = FM.set_interaction([de1, de2], "gated_fusion", output_dim = 64)
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
    while(os.path.exists('./result/'+name+ '/12/'+str(now) + '/')):
        now=now+1
    trainer.metrics_dir = './result/'+name+ '/12/'+str(now) + '/'
    trainer.test(test)

