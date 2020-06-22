# AutoVC

* This code is an implementation of AutoVC. The algorithm is based on the following paper:

```
Qian, K., Zhang, Y., Chang, S., Yang, X., & Hasegawa-Johnson, M. (2019). AutoVC:Zero-shot voice style transfer with only autoencoder loss. arXiv preprint arXiv:1905.05879.
```

* The official code is following:
    * https://github.com/auspicious3000/autovc

* Additional refer:
    * https://github.com/auspicious3000/autovc/issues/33#issuecomment-576881834

# Requirements

* torch >= 1.5.0
* tensorboardX >= 2.0
* librosa >= 0.7.2
* matplotlib >= 3.1.3

* Optional for losses flow
    * tensorboard >= 2.2.2


# Used dataset

* Currently uploaded code is compatible with the following datasets.
* The O mark to the left of the dataset name is the dataset actually used in the uploaded result.

|        | | Dataset   | Dataset address                                 |
|--------|-|-----------|-------------------------------------------------|
| O      | | VCTK      | https://datashare.is.ed.ac.uk/handle/10283/2651 |
| X      | | LibriTTS  | https://openslr.org/60/                         |
| O      | | CMU Arctic| http://www.festvox.org/cmu_arctic/index.html    |
| X      | | VoxCeleb1 | http://www.robots.ox.ac.uk/~vgg/data/voxceleb/  |
| X      | | VoxCeleb2 | http://www.robots.ox.ac.uk/~vgg/data/voxceleb/  |


# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in 'Hyper_Parameter.yaml' according to your environment.

* Sound
    * Setting basic sound parameters.

* Content_Encoder
    * Setting the parameters of content encoder.

* Style_Encoder
    * Setting the parameters of style encoder.
    * Encoder is a pre-trained speaker embedding model.
        * https://github.com/CODEJIN/Speaker_Embedding_Torch
    * All parameters must be matched to pre-trained speaker embedding.

* Decoder
    * Setting the parameters of decoder.

* Postnet
    * Setting the parameters of convolution postnet.

* WaveNet
    * Setting the parameters of Vocoder.
    * This implementation uses a pre-trained Parallel WaveGAN model.
        * https://github.com/CODEJIN/PWGAN_Torch
    * If checkpoint path is `null`, model does not exports wav files.
    * If checkpoint path is not `null`, all parameters must be matched to pre-trained Parallel WaveGAN model.

* Train
    * Setting the parameters of training.
    * When the number of speaekrs in your train dataset is small, I recommend to increase the `Train_Pattern/Accumulated_Dataset_Epoch`.

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Device
    * Setting which GPU device is used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'.


# Generate pattern

## Command
```
python Pattern_Generate.py [parameters]
```

## Parameters

At least, one or more of datasets must be used.

* -vctk <path>
    * Set the path of VCTK. VCTK's patterns are generated.
* -vc1 <path>
    * Set the path of VoxCeleb1. VoxCeleb1's patterns are generated.
* -vc2 <path>
    * Set the path of VoxCeleb2. VoxCeleb2's patterns are generated.
* -libri <path>
    * Set the path of LibriTTS. LibriTTS's patterns are generated.
* -cmua <path>
    * Set the path of CMU Arctic. CMU Arctic's patterns are generated.
* -vc1t <path>
    * Set the path of VoxCeleb1 testset. VoxCeleb1's patterns are generated for an evaluation.
* -mw
    * The number of threads used to create the pattern


# Run

## Command
```
python Train.py -s <int>
```

* `-s <int>`
    * The resume step parameter.
    * Default is 0.
    * When this parameter is 0, model try to find the latest checkpoint in checkpoint path.


# Result

* Current training....

<S>

* Please refer the demo site:
    * https://codejin.github.io/AutoVC_Demo

</S>

# Trained checkpoint

* Current training....

<S>

* This is the checkpoint of ? steps of 2 batchs (? epochs).
* [Checkpoint link](./Example_Results/Checkpoint/S_100000.pkl)
* [Hyperparameter link](./Example_Results/Checkpoint/Hyper_Parameter.yaml)
</S>