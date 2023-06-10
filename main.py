import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
from model import GAN
from train import train_model
from GAN_load import print_results
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool, required=False, help='eval if true train and eval if false')
    parser.add_argument('--model_path', type=str, required=True, help='Model save/load path')
    args = parser.parse_args()
    
    if args.eval == True:
        print_results(args.model_path)
    else:
       train_model()
       print_results(args.model_path) 
