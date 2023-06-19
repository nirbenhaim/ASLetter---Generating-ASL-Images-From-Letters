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
# from data_augmentation import data_augmentation    
from model import GAN
from train import train_model
from util import print_results
import argparse
import os
from import_dataset import download_mnist_sign_language_train_set, download_mnist_sign_language_test_set
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool, required=False, help='eval if true train and eval if false')
    parser.add_argument('--model_path', type=str, required=True, help='Model save/load path')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    args = parser.parse_args()

    # download data
    file_path_train = os.path.join(args.data_path, 'sign_mnist_train.csv')
    if not os.path.exists(file_path_train):
        download_mnist_sign_language_train_set(args.data_path)

    file_path_test = os.path.join(args.data_path, 'sign_mnist_test.csv')
    if not os.path.exists(file_path_test):
        download_mnist_sign_language_test_set(args.data_path)
    
    # run model
    if args.eval == True:
        print_results(args.model_path)
    else:
       train_model(args.data_path, args.model_path)
       print_results(args.model_path)

