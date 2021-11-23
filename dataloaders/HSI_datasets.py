import collections
import math
import os
import random
from socket import MsgFlag

import cv2
import numpy as np
import torch
from torch.utils import data

import scipy
import scipy.ndimage
import scipy.io

# Pavia dataset (Dataset can be downloaded from the following link)
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

class pavia_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["pavia_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["pavia_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["pavia_dataset"]["LR_size"], self.config["pavia_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["pavia_dataset"]["HR_size"], self.config["pavia_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["pavia_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization
            #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))
            #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)
            #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["pavia_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["pavia_dataset"]["max_value"]
        reference   = reference/self.config["pavia_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference

# Botswana dataset
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
class botswana_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        # Determine between train and val splits
        self.split = "train" if is_train else "val"

        #COnfig file
        self.config = config

        #Settings (DIP or no-DIP)
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp = is_dhp

        # Path to botswana dataset
        self.dir = self.config["botswana_dataset"]["data_dir"]

        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")

        # List of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations (Not applicable in this experiment actually)
        self.augmentation = self.config["botswana_dataset"]["augmentation"]

        # Reading the LR crop size
        self.LR_crop_size = (self.config["botswana_dataset"]["LR_size"], self.config["botswana_dataset"]["LR_size"])

        # High resolution crop size
        self.HR_crop_size = [
            self.config["botswana_dataset"]["HR_size"],
            self.config["botswana_dataset"]["HR_size"],
        ]

        # To avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        # Set of all image names
        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # Reading the mat file
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["pavia_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization
            #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))
            #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)
            #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = mat["y"]
            
        # Convert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["botswana_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["botswana_dataset"]["max_value"]
        reference   = reference/self.config["botswana_dataset"]["max_value"] 
       
        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)
            
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference

class chikusei_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        # Settings
        self.split          = "train" if is_train else "val"
        self.config         = config
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp         = is_dhp

        # Paths
        self.dir            = self.config["chikusei_dataset"]["data_dir"]

        # Read train/val image indexes from the text file (train.txt, val.txt, train_dhp.txt)
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")

        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations
        self.augmentation = self.config["chikusei_dataset"]["augmentation"]

        self.LR_crop_size = (self.config["chikusei_dataset"]["LR_size"], self.config["chikusei_dataset"]["LR_size"])

        self.HR_crop_size = [
            self.config["chikusei_dataset"]["HR_size"],
            self.config["chikusei_dataset"]["HR_size"],
        ]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["chikusei_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["chikusei_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["chikusei_dataset"]["max_value"]
        reference   = reference/self.config["chikusei_dataset"]["max_value"]           

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference


###  Loss Angleses Dataset ### 
class la_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        # Settings
        self.split          = "train" if is_train else "val"
        self.config         = config
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp         = is_dhp

        # Paths
        self.dir            = self.config["la_dataset"]["data_dir"]

        # Read train/val image indexes from the text file (train.txt, val.txt, train_dhp.txt)
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")

        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations
        self.augmentation = self.config["la_dataset"]["augmentation"]

        self.LR_crop_size = (self.config["la_dataset"]["LR_size"], self.config["la_dataset"]["LR_size"])

        self.HR_crop_size = [
            self.config["la_dataset"]["HR_size"],
            self.config["la_dataset"]["HR_size"],
        ]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["la_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["la_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["la_dataset"]["max_value"]
        reference   = reference/self.config["la_dataset"]["max_value"]           

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference

###  Botswana (x4) Dataset  ###
class botswana4_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["botswana4_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["botswana4_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["botswana4_dataset"]["LR_size"], self.config["botswana4_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["botswana4_dataset"]["HR_size"], self.config["botswana4_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["botswana4_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization
            #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))
            #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)
            #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["botswana4_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["botswana4_dataset"]["max_value"]
        reference   = reference/self.config["botswana4_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference