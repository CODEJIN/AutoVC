import torch
import numpy as np
import yaml, pickle, os, math
from random import shuffle
#from Pattern_Generator import Pattern_Generate
from Pattern_Generator import Pattern_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Train_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], hp_Dict['Train']['Train_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            # if metadata_Dict['Mel_Length_Dict'][x] > hp_Dict['Train']['Train_Pattern']['Pattern_Length']
            ] * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Mel'], pattern_Dict['Pitch']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Dev_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dev_Dataset, self).__init__()

        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], hp_Dict['Train']['Eval_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List = [
            x for x in metadata_Dict['File_List']
            ]
        self.dataset_Dict = metadata_Dict['Dataset_Dict']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        dataset = self.dataset_Dict[file]
        path = os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], dataset, file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Mel'], pattern_Dict['Pitch']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

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

        source_Label, source_Path, target_Label, target_Path = self.pattern_List[idx]
        _, source_Mel, source_Pitch = Pattern_Generate(source_Path, 15)
        _, target_Mel, _ = Pattern_Generate(target_Path, 15)

        pattern = source_Mel, target_Mel, source_Pitch, source_Label, target_Label

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)

class Collater:
    def __call__(self, batch):
        mels, pitches = zip(*[
            (mel, pitch)
            for mel, pitch in batch
            ])
        
        mels = [
            mel * np.random.uniform(
                low= hp_Dict['Train']['Train_Pattern']['Energy_Variance']['Min'],
                high= hp_Dict['Train']['Train_Pattern']['Energy_Variance']['Max']
                )
            for mel in mels
            ]
            
        stack_Size = np.random.randint(
            low= hp_Dict['Train']['Train_Pattern']['Mel_Length']['Min'],
            high= hp_Dict['Train']['Train_Pattern']['Mel_Length']['Max'] + 1
            )
        contents, pitches = Stack(mels, pitches, size= stack_Size)
        styles = Style_Stack(mels)

        contents = torch.FloatTensor(contents)   # [Batch, Mel_dim, Time]
        styles = torch.FloatTensor(styles)  # [Batch * Samples, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        interpolation_Size = np.random.randint(
            low= int(stack_Size * hp_Dict['Train']['Train_Pattern']['Interpolation']['Min']),
            high= int(stack_Size * hp_Dict['Train']['Train_Pattern']['Interpolation']['Max'])
            )
        interpolation_Size = int(np.ceil(
            interpolation_Size / hp_Dict['Content_Encoder']['Frequency']
            )) * hp_Dict['Content_Encoder']['Frequency']
        contents = torch.nn.functional.interpolate(
            input= contents,
            size= interpolation_Size,
            mode= 'linear',
            align_corners= True
            )
        pitches = torch.nn.functional.interpolate(
            input= pitches.unsqueeze(1),
            size= interpolation_Size,
            mode= 'linear',
            align_corners= True
            ).squeeze(1)
        
        return contents, styles, pitches

class Inference_Collater:
    def __call__(self, batch):
        source_Mels, target_Mels, source_Pitchs, source_Labels, target_Labels = zip(*[
            (source_Mel, target_Mel, source_Pitch, source_Label, target_Label)
            for source_Mel, target_Mel, source_Pitch, source_Label, target_Label in batch
            ])

        contents, pitches, lengths = Stack(source_Mels, source_Pitchs)
        content_Styles = Style_Stack(source_Mels)
        target_Styles = Style_Stack(target_Mels)

        contents = torch.FloatTensor(contents)   # [Batch, Mel_dim, Time]
        content_Styles = torch.FloatTensor(content_Styles)  # [Batch * Samples, Mel_dim, Time]
        target_Styles = torch.FloatTensor(target_Styles)  # [Batch * Samples, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]

        return contents, content_Styles, target_Styles, pitches, lengths, source_Labels, target_Labels


def Stack(mels, pitches, size= None):
    if size is None:
        max_Length = max([mel.shape[0] for mel in mels])
        max_Length = int(np.ceil(
            max_Length / hp_Dict['Content_Encoder']['Frequency']
            )) * hp_Dict['Content_Encoder']['Frequency']
        lengths = [mel.shape[0] for mel in mels]

        mels = np.stack([
            np.pad(mel, [(0, max_Length - mel.shape[0]), (0, 0)], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
            for mel in mels
            ], axis= 0)
        pitches = np.stack([
            np.pad(pitch, (0, max_Length - pitch.shape[0]), constant_values= 0.0)
            for pitch in pitches
            ], axis= 0)
        return mels.transpose(0, 2, 1), pitches, lengths
    
    mel_List = []
    pitch_List = []
    for mel, pitch in zip(mels, pitches):
        if mel.shape[0] > size:
            offset = np.random.randint(0, mel.shape[0] - size)
            mel = mel[offset:offset + size]
            pitch = pitch[offset:offset + size]
        else:
            pad = size - mel.shape[0]
            mel = np.pad(
                mel,
                [[int(np.floor(pad / 2)), int(np.ceil(pad / 2))], [0, 0]],
                mode= 'reflect'
                )
            pitch = np.pad(
                pitch,
                [int(np.floor(pad / 2)), int(np.ceil(pad / 2))],
                mode= 'reflect'
                )                
        mel_List.append(mel)
        pitch_List.append(pitch)

    return np.stack(mel_List, axis= 0).transpose(0, 2, 1), np.stack(pitch_List, axis= 0)

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

    return np.vstack(styles).transpose(0, 2, 1)


if __name__ == "__main__":    
    # dataLoader = torch.utils.data.DataLoader(
    #     dataset= Train_Dataset(),
    #     shuffle= True,
    #     collate_fn= Collater(),
    #     batch_size= hp_Dict['Train']['Batch_Size'],
    #     num_workers= hp_Dict['Train']['Num_Workers'],
    #     pin_memory= True
    #     )

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
    #     speakers, mels, pitches, factors = x
    #     print(speakers.shape)
    #     print(mels.shape)
    #     print(pitches.shape)
    #     print(factors)
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
        speakers, rhymes, contents, pitches, factors, rhyme_Labels, content_Labels, pitch_Labels, lengths = x
        print(speakers.shape)
        print(rhymes.shape)
        print(contents.shape)
        print(pitches.shape)
        print(factors)
        print(rhyme_Labels)
        print(content_Labels)
        print(pitch_Labels)
        print(lengths)
        time.sleep(2.0)







