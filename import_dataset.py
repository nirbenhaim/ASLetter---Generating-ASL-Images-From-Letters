import urllib.request
import os

def download_mnist_sign_language_dataset(save_path):
    train_url = "https://raw.githubusercontent.com/gchilingaryan/Sign-Language/master/sign_mnist_train.csv"
    save_trainset = os.path.join(save_path, "sign_mnist_train.csv")
    
    try:
        urllib.request.urlretrieve(train_url, save_trainset)
        print("MNIST Sign Language train set downloaded successfully!")
    except Exception as e:
        print("An error occurred while downloading the train set:", e)

    test_url = "https://raw.githubusercontent.com/gchilingaryan/Sign-Language/master/sign_mnist_test.csv"
    save_testset = os.path.join(save_path, "sign_mnist_test.csv")
    
    try:
        urllib.request.urlretrieve(test_url, save_testset)
        print("MNIST Sign Language test set downloaded successfully!")
    except Exception as e:
        print("An error occurred while downloading the test set:", e)

# Example usage:
download_mnist_sign_language_dataset(r"C:\path\to\save\directory")