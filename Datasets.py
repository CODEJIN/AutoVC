import torch
import numpy as np
import yaml, librosa, pickle, os

from Pattern_Generator import Pattern_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class StyleDataset(torch.utils.data.Dataset):
    def __init__(self):
        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.fle_List_by_Speaker_Dict = metadata_Dict['File_List_by_Speaker_Dict']
        self.key_List = list(metadata_Dict['File_List_by_Speaker_Dict'].keys())

    def __getitem__(self, idx):
        self.key_List[idx]