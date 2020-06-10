import torch
import yaml, logging

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Content_Encoder(torch.nn.Module):
    def __init__(self):
        super(Content_Encoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        
        self.layer_Dict['Conv'] = torch.nn.Sequential()
        previous_Channels = hp_Dict['Sound']['Mel_Dim'] + hp_Dict['Style_Encoder']['Embedding_Size']
        for index, (channels, kernel_Size) in enumerate(zip(
            hp_Dict['Content_Encoder']['Conv']['Channels'],
            hp_Dict['Content_Encoder']['Conv']['Kernel_Sizes']
            )):
            self.layer_Dict['Conv'].add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.layer_Dict['Conv'].add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= channels
                ))
            self.layer_Dict['Conv'].add_module('ReLU_{}'.format(index), torch.nn.ReLU(
                inplace= True
                ))
            previous_Channels = channels

        self.layer_Dict['BiLSTM'] = torch.nn.LSTM(
            input_size= hp_Dict['Content_Encoder']['Conv']['Channels'][-1],
            hidden_size= hp_Dict['Content_Encoder']['LSTM']['Sizes'],
            num_layers= hp_Dict['Content_Encoder']['LSTM']['Stacks'],
            bias= True,
            batch_first= True,
            bidirectional= True
            )

    def forward(self, mels, styles):
        '''
        mels: [Batch, Mel_dim, Time]
        '''
        self.layer_Dict['BiLSTM'].flatten_parameters()

        if mels.size(2) % hp_Dict['Content_Encoder']['Frequency'] > 0:
            raise ValueError('The frame length of Mel must be a multiple of frequency.')

        x = torch.cat([     # [Batch, Mel_dim + Style_dim, Time]
            mels,
            styles.unsqueeze(2).expand(-1, -1, mels.size(2))
            ], dim= 1)
        
        x = self.layer_Dict['Conv'](x)  # [Batch, Conv_dim, Time]        
        x = self.layer_Dict['BiLSTM'](x.transpose(2, 1))[0].transpose(2, 1)   # [Batch, LSTM_dim * 2, Time]
        x_Forward, x_Backward = x.split(x.size(1) // 2, dim= 1) # [Batch, LSTM_dim, Time] * 2        
        
        x = torch.cat([
            x_Forward[:, :,hp_Dict['Content_Encoder']['Frequency']-1::hp_Dict['Content_Encoder']['Frequency']],
            x_Backward[:, :,::hp_Dict['Content_Encoder']['Frequency']]
            ], dim= 1)  # [Batch, LSTM_dim * 2, Time // Frequency]
        
        return x.repeat_interleave(hp_Dict['Content_Encoder']['Frequency'], dim= 2) # [Batch, LSTM_dim * 2, Time]

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        
        # This part is different between official code and paper.
        # This code is based on the official code.
        self.layer_Dict['Pre_LSTM'] = torch.nn.LSTM(
            input_size= hp_Dict['Content_Encoder']['LSTM']['Sizes'] * 2  + hp_Dict['Style_Encoder']['Embedding_Size'],
            hidden_size= hp_Dict['Decoder']['Pre_LSTM']['Sizes'],
            num_layers= hp_Dict['Decoder']['Pre_LSTM']['Stacks'],
            bias= True,
            batch_first= True
            )
        
        self.layer_Dict['Conv'] = torch.nn.Sequential()        
        previous_Channels = hp_Dict['Decoder']['Pre_LSTM']['Sizes']
        for index, (channels, kernel_Size) in enumerate(zip(
            hp_Dict['Decoder']['Conv']['Channels'],
            hp_Dict['Decoder']['Conv']['Kernel_Sizes']
            )):
            self.layer_Dict['Conv'].add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.layer_Dict['Conv'].add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= channels
                ))
            self.layer_Dict['Conv'].add_module('ReLU_{}'.format(index), torch.nn.ReLU(
                inplace= True
                ))
            previous_Channels = channels

        self.layer_Dict['Post_LSTM'] = torch.nn.LSTM(
            input_size= hp_Dict['Decoder']['Conv']['Channels'][-1],
            hidden_size= hp_Dict['Decoder']['Post_LSTM']['Sizes'],
            num_layers= hp_Dict['Decoder']['Post_LSTM']['Stacks'],
            bias= True,
            batch_first= True
            )
        self.layer_Dict['Linear'] = Linear( #This is same to Conv1x1
            in_features= hp_Dict['Decoder']['Post_LSTM']['Sizes'],
            out_features= hp_Dict['Sound']['Mel_Dim'],
            bias= True
            )
        self.layer_Dict['Postnet'] = Postnet()

    def forward(self, contents, styles):
        '''
        contents: [Batch, Enc_dim, Time]
        styles: [Batch, Style_dim]
        '''
        self.layer_Dict['Pre_LSTM'].flatten_parameters()
        self.layer_Dict['Post_LSTM'].flatten_parameters()

        x = torch.cat([     # [Batch, Enc_dim + Style_dim, Time]
            contents,
            styles.unsqueeze(2).expand(-1, -1, contents.size(2))
            ], dim= 1)
        x, _ = self.layer_Dict['Pre_LSTM'](x.transpose(2, 1))   # [Batch, Time, Pre_LSTM_dim]
        x = self.layer_Dict['Conv'](x.transpose(2, 1))  # [Batch, Conv_dim, Time]
        x, _ = self.layer_Dict['Post_LSTM'](x.transpose(2, 1))   # [Batch, Time, Post_LSTM_dim]
        mels = self.layer_Dict['Linear'](x).transpose(2, 1)    # [Batch, Mel_dim, Time]
        post_Mels = self.layer_Dict['Postnet'](mels= mels)  # [Batch, Mel_dim, Time]

        return mels, post_Mels



class Postnet(torch.nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.layer = torch.nn.Sequential()

        previous_Channels = hp_Dict['Sound']['Mel_Dim']
        for index, (channels, kernel_Size) in enumerate(zip(
            hp_Dict['Postnet']['Channels'] + [hp_Dict['Sound']['Mel_Dim']],
            hp_Dict['Postnet']['Kernel_Sizes'] + [5]
            )):
            self.layer.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_Size,
                padding= (kernel_Size - 1) // 2,
                bias= False,
                w_init_gain= 'tanh'
                ))            
            self.layer.add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= channels
                ))                
            if index < len(hp_Dict['Postnet']['Channels']):
                self.layer.add_module('Tanh_{}'.format(index), torch.nn.Tanh())
            previous_Channels = channels

    def forward(self, mels):
        '''
        mels: [Batch, Mel_dim, Time]
        '''
        return self.layer(mels) + mels


class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'linear', *args, **kwagrs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwagrs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Linear(torch.nn.Linear):
    def __init__(self, w_init_gain= 'linear', *args, **kwagrs):
        self.w_init_gain = w_init_gain
        super(Linear, self).__init__(*args, **kwagrs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


if __name__ == "__main__":
    style_Encoder = Style_Encoder(
        mel_dims= 80,
        lstm_size= 768,
        lstm_stacks= 2,
        embedding_size= 256            
        ).cuda()
    autoVC = AutoVC(style_Encoder).cuda()

    mel = torch.randn(5, 80, 544).cuda()
    style_mel = torch.randn(5, 80, 544).cuda()
    
    x = autoVC(mel, style_mel)


    print(x.shape)
