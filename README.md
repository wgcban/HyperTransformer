# HyperTransformer
Official PyTorch implementation of the paper: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening

# Download datasets

We use three publically available HSI datasets for experiments, namely

1) Pavia Center scene [Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/pavia_centre/Pavia_centre.mat".
2) Botswana [Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/botswana4/Botswana.mat".
3) Chikusei datasets [Download the .mat file here](https://naotoyokoya.com/Download.html), and save it in "./datasets/chikusei/chikusei.mat".


 # Processing the datasets to generate LR-HSI, PAN, and Reference-HR-HSI using Wald's protocol
 We use Wald's protocol to generate LR-HSI and PAN image. To generate those cubic patches,
  1) Run `process_pavia.m` in `./datasets/pavia_centre/` to generate cubic patches. 
  2) Run `process_botswana.m` in `./datasets/botswana4/` to generate cubic patches.
  3) Run `process_chikusei.m` in `./datasets/chikusei/` to generate cubic patches.
 
# Training HyperTransformer 
We use two stage procedure to train our HyperTransformer. We first pre-train HyperTransformer without proposed multi-head attention and then finetune the complete HyperTransformer with multi-head attention.

## Pre-training HyperTransformer
use following codes to pre-train HyperTransformer on the three datasets.
 1) Pre-training on Pavia Center Dataset: Do following changes to config_HSIT_PRE.json: 
    `"experiment_name": "Experiments/HSIT_Pre/"`, 
    `"model": "HSIT_PRE"`, and 
    `"train_dataset": "pavia_dataset"`.
 2) Pre-training on Pavia Center Dataset: Do following changes to config_HSIT_PRE.json: 
    `"experiment_name": "Experiments/HSIT_Pre/"`, 
    `"model": "HSIT_PRE"`, and 
    `"train_dataset": "pavia_dataset"`.
 3) Pre-training on Pavia Center Dataset: Do following changes to config_HSIT_PRE.json: 
    `"experiment_name": "Experiments/HSIT_Pre/"`, 
    `"model": "HSIT_PRE"`, and 
    `"train_dataset": "pavia_dataset"`.
`python train.py --config 'configs/config_HSIT_PRE.json'`



## Fine tuining HyperTransformer


