import matplotlib.pyplot as plt
import torch
from model import GAN

def print_results(model_path):
    # setting
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # Remember that GANs are highly sensitive to hyper-parameters
    random_seed = 123
    LATENT_DIM = 512 # latent vectors dimension [z]
    IMG_SHAPE = (1, 28, 28) # MNIST has 1 color channel, each image 28x8 pixels
    IMG_SIZE = 1
    for x in IMG_SHAPE:
        IMG_SIZE *= x

    # load model
    model = GAN(LATENT_DIM).to(device)
    if torch.cuda.is_available() == False:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))

    # visualization
    model.eval()
    # Make new images
    # z = torch.zeros((5, LATENT_DIM)).uniform_(-1.0, 1.0).to(device)
    z = torch.randn((5, LATENT_DIM)).to(device)
    generated_features = model.generator_forward(z)
    imgs = generated_features.reshape(5, 28, 28)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 2.5))

    for i, ax in enumerate(axes):
        axes[i].imshow(imgs[i].to(torch.device('cpu')).detach(), cmap='binary')

    plt.show()


def wasserstein_loss(y_real, y_fake):
    return abs(torch.mean(y_real) - torch.mean(y_fake))