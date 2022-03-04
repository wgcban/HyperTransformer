# HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening (CVPR'22)

[Wele Gedara Chaminda Bandara](https://www.wgcban.com/), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)


## Introduction
<p align="center">
  <img src="/imgs/HyperTransformer-intro.jpg" width="600" />
</p>

Figure 1. How our **HyperTransformer** differs from existing pansharpening architectures. Traditional pansharpening methods simply concatenate PAN and LR-HSI in **(a)** image domain or **(b)** feature domain to learn the mapping function from LR-HSI to pansharpened HSI. In contrast, **(c)** our HyperTransformer utilizes feature representations of LR-HSI, PAN&#8593;&#8595;, and PAN as Queries (Q), Keys (K), and Values (V) in an attention mechanism to transfer most relevant HR textural features to spectral features of LR-HSI from a backbone network.

## Architecture of our **HyperTransformer**
<p align="center">
  <img src="/imgs/HyperTransformer-Hyper-Transformer.jpg" width="600" />
</p>

Figure 2. Overall  structure  of  the  proposed  HyperTrans-former for textural-spectral feature fusion.

## Complete pansharpening network
<p align="center">
  <img src="/imgs/HyperTransformer-complete_network.jpg" width="1000" />
</p>

Figure 3. The complete pansharperning network. Note that we apply  HyperTransformer at three scales: x1&#8593;, x2&#8593;, and x4&#8593;. RBs denotes the residual blocks.


# Setting up a virtual conda environment
Setup a virtual conda environment using the provided ``environment.yml`` file or ``requirements.txt``.
```
conda env create --name HyperTransformer --file environment.yaml
conda activate HyperTransformer
```
or
```
conda create --name HyperTransformer --file requirements.txt
conda activate HyperTransformer
```

# Download datasets

We use three publically available HSI datasets for experiments, namely

1) **Pavia Center scene** [Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/pavia_centre/Pavia_centre.mat".
2) **Botswana dataset**[Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/botswana4/Botswana.mat".
3) **Chikusei dataset** [Download the .mat file here](https://naotoyokoya.com/Download.html), and save it in "./datasets/chikusei/chikusei.mat".

# Processing the datasets to generate LR-HSI, PAN, and Reference-HR-HSI using Wald's protocol
 We use Wald's protocol to generate LR-HSI and PAN image. To generate those cubic patches,
  1) Run `process_pavia.m` in `./datasets/pavia_centre/` to generate cubic patches. 
  2) Run `process_botswana.m` in `./datasets/botswana4/` to generate cubic patches.
  3) Run `process_chikusei.m` in `./datasets/chikusei/` to generate cubic patches.
 
# Training HyperTransformer 
We use two stage procedure to train our HyperTransformer. 

We first train the backbone of HyperTrasnformer and then fine-tune the MHFA modules. This way we get better results and faster convergence instead of training whole network at once.

## Training the Backbone of HyperTrasnformer
Use the following codes to pre-train HyperTransformer on the three datasets.
 1) Pre-training on Pavia Center Dataset: 
    
    Change "train_dataset" to "pavia_dataset" in config_HSIT_PRE.json. 
    
    Then use following commad to pre-train on Pavia Center dataset.
    `python train.py --config configs/config_HSIT_PRE.json`.
    
 4) Pre-training on Botswana Dataset:
     Change "train_dataset" to "botswana4_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     
 6) Pre-training on Chikusei Dataset: 
     
     Change "train_dataset" to "chikusei_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     

## Fine-tuning the MHFA modules in HyperTrasnformer
Next, we fine-tune the MHFA modules in HyperTransformer starting from pre-trained backbone from the previous step.
 1) Fine-tuning MHFA on Pavia Center Dataset: 

    Change "train_dataset" to "pavia_dataset" in config_HSIT.json. 
    
    Then use the following commad to train HyperTransformer on Pavia Center dataset. 
    
    Please specify path to best model obtained from previous step using --resume.
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/pavia_dataset/N_modules\(4\)/best_model.pth`.
   
 3) Fine-tuning on Botswana Dataset: 

    Change "train_dataset" to "botswana4_dataset" in config_HSIT.json. 
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/botswana4/N_modules\(4\)/best_model.pth`.

 5) Fine-tuning on Chikusei Dataset: 

    Change "train_dataset" to "chikusei_dataset" in config_HSIT.json.
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/chikusei_dataset/N_modules\(4\)/best_model.pth`.
    
# Trained models and pansharpened results on test-set
You can download trained models and final prediction outputs through the follwing links for each dataset.
  1) Pavia Center: [Download here](https://www.dropbox.com/sh/9zg0wrbq6fzx1wa/AACH3mnRlqkVFmo6BF4wcDdaa?dl=0)
  2) Botswana: [Download here](https://www.dropbox.com/sh/e7og46hkn3wuaxr/AACrFOpOSFF2u0hG1CzNYVRxa?dl=0)
  3) Chikusei: [Download here](https://www.dropbox.com/sh/l6gaf723cb6asq4/AABPBUleyZ7aFX8POh_d5jC9a?dl=0)




