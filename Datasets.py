import torch
import numpy as np
import yaml, librosa, pickle, os
from random import sample, shuffle
from itertools import combinations
from Pattern_Generator import Mel_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Train_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], hp_Dict['Train']['Train_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))

        self.file_List_by_Speaker_Dict = {}
        for (dataset, speaker), files in metadata_Dict['File_List_by_Speaker_Dict'].items():
            files = [
                path for path in files
                if metadata_Dict['Mel_Length_Dict'][path] >= hp_Dict['Train']['Train_Pattern']['Mel_Length']
                ]
            if len(files) > 1:
                self.file_List_by_Speaker_Dict[dataset, speaker] = files
        self.key_List = list(self.file_List_by_Speaker_Dict.keys()) * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']

        self.cache_Dict = {}

    def __getitem__(self, idx):
        dataset, speaker = self.key_List[idx]
        files = self.file_List_by_Speaker_Dict[dataset, speaker]
        shuffle(files)  # content and style can be exchanged.
        
        mels = []
        for file in sample(files, 2):
            path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], dataset, file).replace('\\', '/')
            if path in self.cache_Dict.keys():
                mels.append(self.cache_Dict[path])
                continue

            mel = pickle.load(open(path, 'rb'))['Mel']
            mels.append(mel)
            if hp_Dict['Train']['Use_Pattern_Cache']:
                self.cache_Dict[path] = mel
        
        return mels

    def __len__(self):
        return len(self.key_List)

class Dev_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dev_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], hp_Dict['Train']['Eval_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))

        self.file_List_by_Speaker_Dict = metadata_Dict['File_List_by_Speaker_Dict']
        self.key_List = list(self.file_List_by_Speaker_Dict.keys())

        self.cache_Dict = {}

    def __getitem__(self, idx):
        dataset, speaker = self.key_List[idx]
        files = self.file_List_by_Speaker_Dict[dataset, speaker]
        
        mels = []
        for file in sample(files, 2):
            path = os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], dataset, file).replace('\\', '/')
            if path in self.cache_Dict.keys():
                mels.append(self.cache_Dict[path])
                continue

            mel = pickle.load(open(path, 'rb'))['Mel']
            mels.append(mel)
            if hp_Dict['Train']['Use_Pattern_Cache']:
                self.cache_Dict[path] = mel
        
        return mels

    def __len__(self):
        return len(self.key_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path= 'Wav_Path_for_Inference.txt'):
        super(Inference_Dataset, self).__init__()

        self.pattern_List = [
            line.strip().split('\t')
            for line in open(pattern_path, 'r').readlines()[1:]
            ]
        
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]
        
        content_Label, content_Path, style_Label, style_Path = self.pattern_List[idx]
        content_Mel = Mel_Generate(content_Path, 15)
        style_Mel = Mel_Generate(style_Path, 15)
        pattern = content_Mel, style_Mel, content_Label, style_Label

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)



class Collater:
    def __call__(self, batch):
        content_Mels, style_Mels = zip(*[
            (content_Mel, style_Mel)
            for content_Mel, style_Mel in batch
            ])
        content_Style_Mels = torch.FloatTensor(Style_Stack(content_Mels)).transpose(2, 1)   # [Batch, Mel_dim, Time]
        style_Mels = torch.FloatTensor(Style_Stack(style_Mels)).transpose(2, 1)   # [Batch, Mel_dim, Time]
        content_Mels = torch.FloatTensor(Content_Stack(content_Mels)).transpose(2, 1)   # [Batch, Mel_dim, Time]

        return content_Mels, content_Style_Mels, style_Mels

class Inference_Collater:
    def __call__(self, batch):
        content_Mels, style_Mels, content_Mel_Lengths, content_Labels, style_Labels = zip(*[
            (content_Mel, style_Mel, content_Mel.shape[0], content_Label, style_Label)
            for content_Mel, style_Mel, content_Label, style_Label in batch
            ])
        content_Style_Mels = torch.FloatTensor(Style_Stack(content_Mels)).transpose(2, 1)   # [Batch, Mel_dim, Time]
        style_Mels = torch.FloatTensor(Style_Stack(style_Mels)).transpose(2, 1)   # [Batch, Mel_dim, Time]
        content_Mels = torch.FloatTensor(Content_Stack(content_Mels, 3, False)).transpose(2, 1)   # [Batch, Mel_dim, Time]

        return content_Mels, content_Style_Mels, style_Mels, content_Mel_Lengths, content_Labels, style_Labels

def Content_Stack(mels, expands = 1, rand= True):
    mel_List = []
    for mel in mels:
        if mel.shape[0] > hp_Dict['Train']['Train_Pattern']['Mel_Length'] * expands:
            if rand:
                offset = np.random.randint(0, mel.shape[0] - hp_Dict['Train']['Train_Pattern']['Mel_Length'] * expands)
            else:
                offset = 0            
            mel = mel[offset:offset + hp_Dict['Train']['Train_Pattern']['Mel_Length'] * expands]
        else:
            pad = (hp_Dict['Train']['Train_Pattern']['Mel_Length'] * expands - mel.shape[0])
            mel = np.pad(
                mel,
                [[int(np.floor(pad / 2)), int(np.ceil(pad / 2))], [0, 0]] if rand else [[0, pad], [0, 0]],
                mode= 'reflect'
                )
        mel_List.append(mel)
    return np.stack(mel_List)

def Style_Stack(mels):
    styles = []
    for mel in mels:
        overlap_Length = hp_Dict['Style_Encoder']['Inference']['Overlap_Length']
        slice_Length = hp_Dict['Style_Encoder']['Inference']['Slice_Length']
        required_Length = hp_Dict['Style_Encoder']['Inference']['Samples'] * (slice_Length - overlap_Length) + overlap_Length

        if mel.shape[0] > required_Length:
            offset = np.random.randint(0, mel.shape[0] - required_Length)
            mel = mel[offset:offset + required_Length]
        else:
            pad = (required_Length - mel.shape[0]) / 2
            mel = np.pad(
                mel,
                [[int(np.floor(pad)), int(np.ceil(pad))], [0, 0]],
                mode= 'reflect'
                )

        mel = np.stack([
            mel[index:index + slice_Length]
            for index in range(0, required_Length - overlap_Length, slice_Length - overlap_Length)
            ])
        styles.append(mel)

    return np.vstack(styles)


if __name__ == "__main__":    
    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= Train_Dataset(),
    #     shuffle= True,
    #     collate_fn= Collater(),
    #     batch_size= hp_Dict['Train']['Batch_Size'],
    #     num_workers= hp_Dict['Train']['Num_Workers'],
    #     pin_memory= True
    #     )

    # import time
    # for x in dataLoader:
    #     content_Mels, content_Style_Mels, style_Mels = x
    #     print(content_Mels.shape)
    #     print(content_Style_Mels.shape)
    #     print(style_Mels.shape)
    #     time.sleep(2.0)

    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= Dev_Dataset(),
    #     shuffle= True,
    #     collate_fn= Collater(),
    #     batch_size= hp_Dict['Train']['Batch_Size'],
    #     num_workers= hp_Dict['Train']['Num_Workers'],
    #     pin_memory= True
    #     )

    # import time
    # for x in dataLoader:
    #     content_Mels, content_Style_Mels, style_Mels = x
    #     print(content_Mels.shape)
    #     print(content_Style_Mels.shape)
    #     print(style_Mels.shape)
    #     time.sleep(2.0)

    dataLoader = torch.utils.data.DataLoader(
        dataset= Inference_Dataset(),
        shuffle= False,
        collate_fn= Inference_Collater(),
        batch_size= hp_Dict['Train']['Batch_Size'],
        num_workers= hp_Dict['Train']['Num_Workers'],
        pin_memory= True
        )

    import time
    for x in dataLoader:
        content_Mels, content_Style_Mels, style_Mels, content_Labels, style_Labels = x
        print(content_Mels.shape)
        print(content_Style_Mels.shape)
        print(style_Mels.shape)
        print(content_Labels)
        print(style_Labels)
        print(content_Mels[0])
        print(content_Mels[3])
        print(content_Mels[6])
        print(content_Mels[9])
        time.sleep(2.0)
