from train import train_model
from util import print_results
import argparse
import os
from import_dataset import download_mnist_sign_language_train_set, download_mnist_sign_language_test_set
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool, required=False, help='Model evaluation if true. Model training and evaluating if false')
    parser.add_argument('--model_path', type=str, required=True, help='Model save/load directory')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset save/load directory')
    parser.add_argument('--chosen_letter', type=str, required=False, help='Letter chosen to be printed, only when purely evaluating')
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
        print_results(args.model_path, chosen_letter=args.chosen_letter)
    else:
       train_model(args.data_path, args.model_path)
       print_results(args.model_path)

