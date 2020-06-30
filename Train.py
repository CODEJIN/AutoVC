import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample

from Modules import Content_Encoder, Decoder
from Datasets import Train_Dataset, Dev_Dataset, Inference_Dataset, Collater, Inference_Collater

from Style_Encoder.Modules import Encoder as Style_Encoder, Normalize
from PWGAN.Modules import Generator as PWGAN

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.loss_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer = SummaryWriter(hp_Dict['Log_Path'])

        self.Style_Encoder_Load_Checkpoint()

        if not hp_Dict['WaveNet']['Checkpoint_Path'] is None:
            self.PWGAN_Load_Checkpoint()
        
        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Train_Dataset()
        dev_Dataset = Dev_Dataset()
        inference_Dataset = Inference_Dataset()
        logging.info('The number of train speakers = {}.'.format(len(train_Dataset) // hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']))
        logging.info('The number of development speakers = {}.'.format(len(dev_Dataset)))
        logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater()
        inference_Collater = Inference_Collater()

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= False,
            collate_fn= inference_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        
    def Model_Generate(self):
        self.model_Dict = {
            'Content_Encoder': Content_Encoder().to(device),
            'Style_Encoder': Style_Encoder(
                mel_dims= hp_Dict['Sound']['Mel_Dim'],
                lstm_size= hp_Dict['Style_Encoder']['LSTM']['Sizes'],
                lstm_stacks= hp_Dict['Style_Encoder']['LSTM']['Stacks'],
                embedding_size= hp_Dict['Style_Encoder']['Embedding_Size'],
                ).to(device),
            'Decoder': Decoder().to(device)
            }
        self.criterion_Dict = {
            'MAE': torch.nn.L1Loss().to(device),
            'MSE': torch.nn.MSELoss().to(device)
            }
        self.optimizer = torch.optim.Adam(
            params= list(self.model_Dict['Content_Encoder'].parameters()) + list(self.model_Dict['Decoder'].parameters()),
            lr= hp_Dict['Train']['Learning_Rate']['Initial'],
            betas=(hp_Dict['Train']['ADAM']['Beta1'], hp_Dict['Train']['ADAM']['Beta2']),
            eps= hp_Dict['Train']['ADAM']['Epsilon'],
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer= self.optimizer,
            step_size= hp_Dict['Train']['Learning_Rate']['Decay_Step'],
            gamma= hp_Dict['Train']['Learning_Rate']['Decay_Rate'],
            )

        if hp_Dict['Use_Mixed_Precision']:
            try:
                from apex import amp
                (self.model_Dict['Content_Encoder'], self.model_Dict['Style_Encoder'], self.model_Dict['Decoder']), self.optimizer = amp.initialize(
                    models= [self.model_Dict['Content_Encoder'], self.model_Dict['Style_Encoder'], self.model_Dict['Decoder']],
                    optimizers=self.optimizer
                    )
            except:
                logging.info('There is no apex modules in the environment. Mixed precision does not work.')

        logging.info(self.model_Dict['Content_Encoder'])
        logging.info(self.model_Dict['Style_Encoder'])
        logging.info(self.model_Dict['Decoder'])


    def Train_Step(self, content_Mels, content_Style_Mels, style_Mels):
        loss_Dict = {}

        content_Mels = content_Mels.to(device)
        content_Style_Mels = content_Style_Mels.to(device)
        style_Mels = style_Mels.to(device)
        
        with torch.no_grad():
            content_Styles = self.model_Dict['Style_Encoder'](content_Style_Mels)
            styles = self.model_Dict['Style_Encoder'](style_Mels)
            content_Styles = Normalize(content_Styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])
            styles = Normalize(styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])
        
        contents = self.model_Dict['Content_Encoder'](
            mels= content_Mels,
            styles= content_Styles
            )
        pre_Mels, post_Mels = self.model_Dict['Decoder'](
            contents= contents,
            styles= styles
            )
        
        recontructed_Styles = self.model_Dict['Style_Encoder'](post_Mels[:, :, :hp_Dict['Style_Encoder']['Inference']['Slice_Length']])        
        recontructed_Styles = torch.nn.functional.normalize(recontructed_Styles, p=2, dim= 1)
        reconstructed_Contents = self.model_Dict['Content_Encoder'](
            mels= post_Mels,
            styles= recontructed_Styles
            )

        loss_Dict['Pre_Reconstructed'] = \
            hp_Dict['Train']['Loss_Weight']['Pre_Mel_L1'] * self.criterion_Dict['MAE'](pre_Mels, content_Mels) + \
            hp_Dict['Train']['Loss_Weight']['Pre_Mel_L2'] * self.criterion_Dict['MSE'](pre_Mels, content_Mels)
        loss_Dict['Post_Reconstructed'] = \
            hp_Dict['Train']['Loss_Weight']['Post_Mel_L1'] * self.criterion_Dict['MAE'](post_Mels, content_Mels) + \
            hp_Dict['Train']['Loss_Weight']['Post_Mel_L2'] * self.criterion_Dict['MSE'](post_Mels, content_Mels)
        loss_Dict['Content'] = \
            hp_Dict['Train']['Loss_Weight']['Content_L1'] * self.criterion_Dict['MAE'](reconstructed_Contents, contents) + \
            hp_Dict['Train']['Loss_Weight']['Content_L2'] * self.criterion_Dict['MSE'](reconstructed_Contents, contents)
        loss_Dict['Total'] = loss_Dict['Pre_Reconstructed'] + loss_Dict['Post_Reconstructed'] + loss_Dict['Content']

        self.optimizer.zero_grad()                
        loss_Dict['Total'].backward()        
        torch.nn.utils.clip_grad_norm_(
            parameters= list(self.model_Dict['Content_Encoder'].parameters()) + list(self.model_Dict['Decoder'].parameters()),
            max_norm= hp_Dict['Train']['Gradient_Norm']
            )        
        self.optimizer.step()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Train'][tag] += loss

    def Train_Epoch(self):
        for content_Mels, content_Style_Mels, style_Mels in self.dataLoader_Dict['Train']:
            self.Train_Step(content_Mels, content_Style_Mels, style_Mels)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.loss_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.loss_Dict['Train'].items()
                    }
                self.Write_to_Tensorboard('Train', self.loss_Dict['Train'])
                self.loss_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()

            if self.steps % hp_Dict['Train']['Inference_Interval'] == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']

    @torch.no_grad()    
    def Evaluation_Step(self, content_Mels, content_Style_Mels, style_Mels):
        loss_Dict = {}

        content_Mels = content_Mels.to(device)
        content_Style_Mels = content_Style_Mels.to(device)
        style_Mels = style_Mels.to(device)
        
        content_Styles = self.model_Dict['Style_Encoder'](content_Style_Mels)
        styles = self.model_Dict['Style_Encoder'](style_Mels)
        content_Styles = Normalize(content_Styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])
        styles = Normalize(styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])

        contents = self.model_Dict['Content_Encoder'](
            mels= content_Mels,
            styles= content_Styles
            )
        pre_Mels, post_Mels = self.model_Dict['Decoder'](
            contents= contents,
            styles= styles
            )

        recontructed_Styles = self.model_Dict['Style_Encoder'](post_Mels)
        reconstructed_Contents = self.model_Dict['Content_Encoder'](
            mels= post_Mels,
            styles= recontructed_Styles
            )

        loss_Dict['Pre_Reconstructed'] = \
            hp_Dict['Train']['Loss_Weight']['Pre_Mel_L1'] * self.criterion_Dict['MAE'](pre_Mels, content_Mels) + \
            hp_Dict['Train']['Loss_Weight']['Pre_Mel_L2'] * self.criterion_Dict['MSE'](pre_Mels, content_Mels)
        loss_Dict['Post_Reconstructed'] = \
            hp_Dict['Train']['Loss_Weight']['Post_Mel_L1'] * self.criterion_Dict['MAE'](post_Mels, content_Mels) + \
            hp_Dict['Train']['Loss_Weight']['Post_Mel_L2'] * self.criterion_Dict['MSE'](post_Mels, content_Mels)
        loss_Dict['Content'] = \
            hp_Dict['Train']['Loss_Weight']['Content_L1'] * self.criterion_Dict['MAE'](reconstructed_Contents, contents) + \
            hp_Dict['Train']['Loss_Weight']['Content_L2'] * self.criterion_Dict['MSE'](reconstructed_Contents, contents)
        loss_Dict['Total'] = loss_Dict['Pre_Reconstructed'] + loss_Dict['Post_Reconstructed'] + loss_Dict['Content']

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Evaluation'][tag] += loss
    
    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (content_Mels, content_Style_Mels, style_Mels) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Evaluation_Step(content_Mels, content_Style_Mels, style_Mels)

        self.loss_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.loss_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.loss_Dict['Evaluation'])
        self.loss_Dict['Evaluation'] = defaultdict(float)

        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Inference_Step(self, content_Mels, content_Style_Mels, style_Mels, content_Mel_Lengths, content_Labels, style_Labels, start_Index= 0, tag_Step= False, tag_Index= False):
        content_Mels = content_Mels.to(device)
        content_Style_Mels = content_Style_Mels.to(device)
        style_Mels = style_Mels.to(device)
        
        content_Styles = self.model_Dict['Style_Encoder'](content_Style_Mels)
        styles = self.model_Dict['Style_Encoder'](style_Mels)
        content_Styles = Normalize(content_Styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])
        styles = Normalize(styles, samples= hp_Dict['Style_Encoder']['Inference']['Samples'])

        contents = self.model_Dict['Content_Encoder'](
            mels= content_Mels,
            styles= content_Styles
            )
        _, post_Mels = self.model_Dict['Decoder'](
            contents= contents,
            styles= styles
            )

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps)).replace("\\", "/"), exist_ok= True)

        for index, (content_Mel, post_Mel, content_Mel_Length, content_Label, style_Label) in enumerate(zip(
            content_Mels.cpu().numpy(),
            post_Mels.cpu().numpy(),
            content_Mel_Lengths,
            content_Labels,
            style_Labels
            )):
            new_Figure = plt.figure(figsize=(16, 8 * 2), dpi=100)
            plt.subplot(311)
            plt.imshow(content_Mel[:, :content_Mel_Length], aspect='auto', origin='lower')
            plt.title('Original    Index: {}    Original: {}    ->    Conversion: {}'.format(index + start_Index, content_Label, style_Label))            
            plt.colorbar()
            plt.subplot(312)
            plt.imshow(post_Mel[:, :content_Mel_Length], aspect='auto', origin='lower')
            plt.title('Conversion    Index: {}    Original: {}    ->    Conversion: {}'.format(index + start_Index, content_Label, style_Label))
            plt.colorbar()
            plt.subplot(313)
            plt.imshow(content_Mel[:, :content_Mel_Length] - post_Mel[:, :content_Mel_Length], aspect='auto', origin='lower')
            plt.title('Difference    Index: {}    Original: {}    ->    Conversion: {}'.format(index + start_Index, content_Label, style_Label))
            plt.colorbar()
            plt.tight_layout()
            file = '{}C_{}.S_{}{}.PNG'.format(
                'Step-{}.'.format(self.steps) if tag_Step else '',
                content_Label,
                style_Label,
                '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                )
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), file).replace("\\", "/")
                )
            plt.close(new_Figure)

        if 'PWGAN' in self.model_Dict.keys():
            noises = torch.randn(post_Mels.size(0), post_Mels.size(2) * hp_Dict['Sound']['Frame_Shift']).to(device)                
            post_Mels = torch.nn.functional.pad(
                post_Mels,
                pad= (hp_Dict['WaveNet']['Upsample']['Pad'], hp_Dict['WaveNet']['Upsample']['Pad']),
                mode= 'replicate'
                )
            content_Mels = torch.nn.functional.pad(
                content_Mels,
                pad= (hp_Dict['WaveNet']['Upsample']['Pad'], hp_Dict['WaveNet']['Upsample']['Pad']),
                mode= 'replicate'
                )

            for index, (audio, mel_Length, content_Label, style_Label) in enumerate(zip(
                self.model_Dict['PWGAN'](noises, post_Mels).cpu().numpy(),
                content_Mel_Lengths,
                content_Labels,
                style_Labels
                )):
                file = '{}C_{}.S_{}{}.Conversion.WAV'.format(
                    'Step-{}.'.format(self.steps) if tag_Step else '',
                    content_Label,
                    style_Label,
                    '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                    )
                wavfile.write(
                    filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), file).replace("\\", "/"),
                    data= (audio[:mel_Length * hp_Dict['Sound']['Frame_Shift']] * 32767.5).astype(np.int16),
                    rate= hp_Dict['Sound']['Sample_Rate']
                    )
            for index, (audio, mel_Length, content_Label, style_Label) in enumerate(zip(
                self.model_Dict['PWGAN'](noises, content_Mels).cpu().numpy(),
                content_Mel_Lengths,
                content_Labels,
                style_Labels
                )):
                file = '{}C_{}.S_{}{}.Original.WAV'.format(
                    'Step-{}.'.format(self.steps) if tag_Step else '',
                    content_Label,
                    style_Label,
                    '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                    )
                wavfile.write(
                    filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), file).replace("\\", "/"),
                    data= (audio[:mel_Length * hp_Dict['Sound']['Frame_Shift']] * 32767.5).astype(np.int16),
                    rate= hp_Dict['Sound']['Sample_Rate']
                    )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (content_Mels, content_Style_Mels, style_Mels, content_Mel_Lengths, content_Labels, style_Labels) in tqdm(
            enumerate(self.dataLoader_Dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Inference_Step(content_Mels, content_Style_Mels, style_Mels, content_Mel_Lengths, content_Labels, style_Labels, start_Index= step * hp_Dict['Train']['Batch_Size'])

        for model in self.model_Dict.values():
            model.train()


    def Style_Encoder_Load_Checkpoint(self):
        state_Dict = torch.load(
            hp_Dict['Style_Encoder']['Checkpoint_Path'],
            map_location= 'cpu'
            )
        self.model_Dict['Style_Encoder'].load_state_dict(state_Dict['Model'])

        logging.info('Speaker embedding checkpoint \'{}\' loaded.'.format(hp_Dict['Style_Encoder']['Checkpoint_Path']))

    def Load_Checkpoint(self):
        if self.steps == 0:
            path = None
            for root, _, files in os.walk(hp_Dict['Checkpoint_Path']):
                path = max(
                    [os.path.join(root, file).replace('\\', '/') for file in files],
                    key = os.path.getctime
                    )
                break
            if path is None:
                return  # Initial training
        else:
            path = os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        self.model_Dict['Content_Encoder'].load_state_dict(state_Dict['Model']['Content_Encoder'])
        self.model_Dict['Decoder'].load_state_dict(state_Dict['Model']['Decoder'])        
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'Content_Encoder': self.model_Dict['Content_Encoder'].state_dict(),
                'Decoder': self.model_Dict['Decoder'].state_dict()
                },
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),            
            'Steps': self.steps,
            'Epochs': self.epochs,
            }

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def PWGAN_Load_Checkpoint(self):
        self.model_Dict['PWGAN'] = PWGAN().to(device)

        if hp_Dict['Use_Mixed_Precision']:
            try:
                from apex import amp
                self.model_Dict['PWGAN'] = amp.initialize(
                    models=self.model_Dict['PWGAN']
                    )
            except:
                pass

        state_Dict = torch.load(
            hp_Dict['WaveNet']['Checkpoint_Path'],
            map_location= 'cpu'
            )
        self.model_Dict['PWGAN'].load_state_dict(state_Dict['Model']['Generator'])

        logging.info('PWGAN checkpoint \'{}\' loaded.'.format(hp_Dict['WaveNet']['Checkpoint_Path']))


    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )
        
        hp_Path = os.path.join(hp_Dict['Checkpoint_Path'], 'Hyper_Parameter.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)
            yaml.dump(hp_Dict, open(hp_Path, 'w'))

        if hp_Dict['Train']['Initial_Inference']:
            self.Evaluation_Epoch()
            self.Inference_Epoch()

        for model in self.model_Dict.values():
            model.train()

        while self.steps < hp_Dict['Train']['Max_Step']:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, loss_Dict):
        for tag, loss in loss_Dict.items():
            self.writer.add_scalar('{}/{}'.format(category, tag), loss, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)
    new_Trainer.Train()