import numpy as np
import yaml, os, pickle, librosa, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as PE
from threading import Thread
from random import shuffle

from Audio import Audio_Prep, Mel_Generate

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
top_DB_Dict = {'VCTK': 15, 'VC1': 23, 'VC2': 23, 'Libri': 23, 'CMUA': 60}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'

def Pattern_File_Generate(path, speaker, dataset, tag= '', eval= False):
    pattern_Path = hp_Dict['Train']['Eval_Pattern' if eval else 'Train_Pattern']['Path']

    pickle_Path = '{}.{}{}.{}.PICKLE'.format(
        dataset,
        speaker,
        '.{}'.format(tag) if tag != '' else tag,
        os.path.splitext(os.path.basename(path))[0]        
        )
    pickle_Path = os.path.join(pattern_Path, dataset, pickle_Path).replace("\\", "/")

    if os.path.exists(pickle_Path):
        return

    os.makedirs(os.path.join(pattern_Path, dataset).replace('\\', '/'), exist_ok= True)    
    try:
        new_Pattern_Dict = {
            'Mel': Mel_Generate(
                audio= Audio_Prep(path, hp_Dict['Sound']['Sample_Rate']),
                sample_rate= hp_Dict['Sound']['Sample_Rate'],
                num_frequency= hp_Dict['Sound']['Spectrogram_Dim'],
                num_mel= hp_Dict['Sound']['Mel_Dim'],
                window_length= hp_Dict['Sound']['Frame_Length'],
                hop_length= hp_Dict['Sound']['Frame_Shift'],        
                mel_fmin= hp_Dict['Sound']['Mel_F_Min'],
                mel_fmax= hp_Dict['Sound']['Mel_F_Max'],
                max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
                ),
            'Speaker': speaker,
            'Dataset': dataset,
            }
    except Exception as e:
        print('Error: {} in {}'.format(e, path))
        return
    
    with open(pickle_Path, 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=4)
            
def VCTK_Info_Load(vctk_Path):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    try:
        with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
            vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    except:
        vctk_Non_Outlier_List = None

    vctk_File_Path_List = []
    for root, _, files in os.walk(vctk_Wav_Path):
        for file in files:
            if not vctk_Non_Outlier_List is None and not file in vctk_Non_Outlier_List:
                continue
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue

            vctk_File_Path_List.append(wav_File_Path)
    
    vctk_Speaker_Dict = {
        path: path.split('/')[-2].upper()
        for path in vctk_File_Path_List
        }

    print('VCTK info generated: {}'.format(len(vctk_File_Path_List)))
    return vctk_File_Path_List, vctk_Speaker_Dict

def VC1_Info_Load(vc1_Path):
    vc1_File_Path_List = []
    for root, _, files in os.walk(vc1_Path):
        for file in files:
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            vc1_File_Path_List.append(wav_File_Path)
    
    vc1_Speaker_Dict = {
        path: path.split('/')[-3].upper()
        for path in vc1_File_Path_List
        }

    print('VC1 info generated: {}'.format(len(vc1_File_Path_List)))
    return vc1_File_Path_List, vc1_Speaker_Dict

def VC2_Info_Load(vc2_Path):
    vc2_File_Path_List = []
    for root, _, files in os.walk(vc2_Path):
        for file in files:
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            vc2_File_Path_List.append(wav_File_Path)
    
    vc2_Speaker_Dict = {
        path: path.split('/')[-3].upper()
        for path in vc2_File_Path_List
        }

    print('VC2 info generated: {}'.format(len(vc2_File_Path_List)))
    return vc2_File_Path_List, vc2_Speaker_Dict


def Libri_Info_Load(libri_Path):
    libri_File_Path_List = []
    for root, _, files in os.walk(libri_Path):
        for file in files:
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            libri_File_Path_List.append(wav_File_Path)
            
    libri_Speaker_Dict = {
        path: path.split('/')[-3].upper()
        for path in libri_File_Path_List
        }

    print('Libri info generated: {}'.format(len(libri_File_Path_List)))
    return libri_File_Path_List, libri_Speaker_Dict

def CMUA_Info_Load(cmua_Path):
    cmua_File_Path_List = []
    for root, _, files in os.walk(cmua_Path):
        for file in files:
            wav_File_Path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            cmua_File_Path_List.append(wav_File_Path)
            
    cmua_Speaker_Dict = {
        path: path.split('/')[-3].split('_')[2].upper()
        for path in cmua_File_Path_List
        }

    print('CMUA info generated: {}'.format(len(cmua_File_Path_List)))
    return cmua_File_Path_List, cmua_Speaker_Dict

def Metadata_Generate(eval= False):
    pattern_Path = hp_Dict['Train']['Eval_Pattern' if eval else 'Train_Pattern']['Path']

    new_Metadata_Dict = {
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],
        'File_List': [],        
        'Mel_Length_Dict': {},
        'Dataset_Dict': {},
        'Speaker_Dict': {},
        'File_List_by_Speaker_Dict': {},
        }

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                    pattern_Dict = pickle.load(f)
                # try:
                    if not all([key in pattern_Dict.keys() for key in ('Mel', 'Speaker', 'Dataset')]):
                        continue

                    new_Metadata_Dict['File_List'].append(file)
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['Speaker_Dict'][file] = pattern_Dict['Speaker']
                    if not (pattern_Dict['Dataset'], pattern_Dict['Speaker']) in new_Metadata_Dict['File_List_by_Speaker_Dict'].keys():
                        new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Dataset'], pattern_Dict['Speaker']] = []
                    new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Dataset'], pattern_Dict['Speaker']].append(file)
                # except:
                #     print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
                
    with open(os.path.join(pattern_Path, hp_Dict['Train']['Train_Pattern']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=4)

    print('Metadata generate done.')

def Metadata_Subset_Generate(base_Metadata_Path, subset_Metadata_Name, use_Datsets):
    assert os.path.basename(base_Metadata_Path) != subset_Metadata_Name, 'Subset metadata name must be different from base metadata name.'

    metadata_Dict = pickle.load(open(base_Metadata_Path, 'rb'))

    metadata_Dict['File_List'] = [
        file for file in metadata_Dict['File_List']
        if metadata_Dict['Dataset_Dict'][file] in use_Datsets
        ]
    metadata_Dict['Mel_Length_Dict'] = {
        file: length for file, length in metadata_Dict['Mel_Length_Dict'].items()
        if metadata_Dict['Dataset_Dict'][file] in use_Datsets
        }
    metadata_Dict['Speaker_Dict'] = {
        file: speaker for file, speaker in metadata_Dict['Speaker_Dict'].items()
        if metadata_Dict['Dataset_Dict'][file] in use_Datsets
        }
    metadata_Dict['File_List_by_Speaker_Dict'] = {
        (dataset, speaker): file_List for (dataset, speaker), file_List in metadata_Dict['File_List_by_Speaker_Dict'].items()
        if dataset in use_Datsets
        }
    metadata_Dict['Dataset_Dict'] = {
        file: dataset for file, dataset in metadata_Dict['Dataset_Dict'].items()
        if metadata_Dict['Dataset_Dict'][file] in use_Datsets
        }

    with open(os.path.join(os.path.dirname(base_Metadata_Path), subset_Metadata_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(metadata_Dict, f, protocol=4)

    print('Metadata subset generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-vc1", "--vc1_path", required=False)
    argParser.add_argument("-vc2", "--vc2_path", required=False)
    argParser.add_argument("-libri", "--libri_path", required=False)
    argParser.add_argument("-cmua", "--cmua_path", required=False)

    argParser.add_argument("-vc1t", "--vc1_test_path", required=False)

    argParser.add_argument("-mw", "--max_worker", default= 10, type= int)
    args = argParser.parse_args()
    
    path_List = []
    speaker_Dict = {}
    dataset_Dict = {}
    tag_Dict = {}
    if not args.vctk_path is None:
        vctk_File_Path_List, vctk_Speaker_Dict = VCTK_Info_Load(vctk_Path= args.vctk_path)
        path_List.extend(vctk_File_Path_List)
        speaker_Dict.update(vctk_Speaker_Dict)
        dataset_Dict.update({path: 'VCTK' for path in vctk_File_Path_List})
        tag_Dict.update({path: '' for path in vctk_File_Path_List})
    if not args.vc1_path is None:
        vc1_File_Path_List, vc1_Speaker_Dict = VC1_Info_Load(vc1_Path= args.vc1_path)
        path_List.extend(vc1_File_Path_List)
        speaker_Dict.update(vc1_Speaker_Dict)
        dataset_Dict.update({path: 'VC1' for path in vc1_File_Path_List})
        tag_Dict.update({path: '{}'.format(path.split('/')[-2]) for path in vc1_File_Path_List})
    if not args.vc2_path is None:
        vc2_File_Path_List, vc2_Speaker_Dict = VC2_Info_Load(vc2_Path= args.vc2_path)
        path_List.extend(vc2_File_Path_List)
        speaker_Dict.update(vc2_Speaker_Dict)
        dataset_Dict.update({path: 'VC2' for path in vc2_File_Path_List})
        tag_Dict.update({path: '{}'.format(path.split('/')[-2]) for path in vc2_File_Path_List})
    if not args.libri_path is None:
        libri_File_Path_List, libri_Speaker_Dict = Libri_Info_Load(libri_Path= args.libri_path)
        path_List.extend(libri_File_Path_List)
        speaker_Dict.update(libri_Speaker_Dict)
        dataset_Dict.update({path: 'Libri' for path in libri_File_Path_List})
        tag_Dict.update({path: '' for path in libri_File_Path_List})
    if not args.cmua_path is None:
        cmua_File_Path_List, cmua_Speaker_Dict = CMUA_Info_Load(cmua_Path= args.cmua_path)
        path_List.extend(cmua_File_Path_List)
        speaker_Dict.update(cmua_Speaker_Dict)
        dataset_Dict.update({path: 'CMUA' for path in cmua_File_Path_List})
        tag_Dict.update({path: '' for path in cmua_File_Path_List})

    if len(path_List) == 0:
        raise ValueError('Total info count must be bigger than 0.')
    print('Total info generated: {}'.format(len(path_List)))

    # with PE(max_workers = args.max_worker) as pe:
    #     for _ in tqdm(
    #         pe.map(
    #             lambda params: Pattern_File_Generate(*params),
    #             [(path, speaker_Dict[path], dataset_Dict[path], tag_Dict[path], False) for path in path_List]
    #             ),
    #         total= len(path_List)
    #         ):
    #         pass
    Metadata_Generate()

    if not args.vc1_test_path is None:
        path_List, speaker_Dict = VC1_Info_Load(vc1_Path= args.vc1_test_path)
        dataset_Dict = {path: 'VC1' for path in path_List}
        tag_Dict = {path: '.{}'.format(path.split('/')[-2]) for path in path_List}

        with PE(max_workers = args.max_worker) as pe:
            for _ in tqdm(
                pe.map(
                    lambda params: Pattern_File_Generate(*params),
                    [(path, speaker_Dict[path], dataset_Dict[path], tag_Dict[path], True) for path in path_List]
                    ),
                total= len(path_List)
                ):
                pass

        Metadata_Generate(eval= True)