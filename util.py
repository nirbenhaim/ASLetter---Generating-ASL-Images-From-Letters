import matplotlib.pyplot as plt
import torch
from model import Generator
import os

CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def print_results(model_path, chosen_letter=None):
    # setting
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # Remember that GANs are highly sensitive to hyper-parameters
    random_seed = 123
    BATCH_SIZE = 128
    LATENT_DIM = 100
    NUM_CLASSES = 26
    NUM_EPOCHS = 30
    PRINT_NUM = 24

    # load model
    generator = Generator(LATENT_DIM, NUM_CLASSES).to(device)
    gen_save_path = os.path.join(model_path, 'generator.pth')
    if torch.cuda.is_available() == False:
        generator.load_state_dict(torch.load(gen_save_path, map_location=torch.device('cpu')))
    else:
        generator.load_state_dict(torch.load(gen_save_path))

    # visualization
    generator.eval()

    # Make new images
    # z = torch.randn((PRINT_NUM, LATENT_DIM)).to(device)
    # fake_labels = torch.LongTensor(5).random_(0, NUM_CLASSES-2)
    # fake_labels_reshape = torch.empty((PRINT_NUM, NUM_CLASSES))
    # for idx, fake_label in enumerate(fake_labels):
    #     fake_labels_temp = torch.zeros(1, NUM_CLASSES).squeeze()
    #     if fake_label == 9:
    #         fake_label = 24
    #     fake_labels_temp[fake_label] = 1.
    #     fake_labels_reshape[idx] = fake_labels_temp

    # fake_labels_reshape = fake_labels_reshape.to(device)

    if chosen_letter is None:
        z = torch.randn((PRINT_NUM, LATENT_DIM)).to(device)
        fake_labels = torch.linspace(0, PRINT_NUM-1, steps=PRINT_NUM)
        fake_labels_reshape = torch.empty((PRINT_NUM, NUM_CLASSES))
        for idx, fake_label in enumerate(fake_labels):
            fake_labels_temp = torch.zeros(1, NUM_CLASSES).squeeze()
            if fake_label >= 9.:
                fake_label += 1.
            fake_labels_temp[int(fake_label)] = 1.
            fake_labels_reshape[idx] = fake_labels_temp

        fake_labels_reshape = fake_labels_reshape.to(device)

        generated_features = generator.forward(z, fake_labels_reshape)
        imgs = generated_features.reshape(PRINT_NUM, 28, 28)

        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 2.5))

        for i in range(4):
            for j in range(6):
                axes[i][j].imshow(imgs[i*6+j].to(torch.device('cpu')).detach(), cmap='binary')
                axes[i][j].title.set_text(CLASS_LIST[int(fake_labels_reshape[i*6+j].argmax())])

        plt.show()

    else:
        z = torch.randn((PRINT_NUM, LATENT_DIM)).to(device)

        chosen_idx = 0
        for idx, letter in enumerate(CLASS_LIST):
            if letter == chosen_letter.upper():
                chosen_idx = idx

        if CLASS_LIST[chosen_idx] in ['J', 'Z']:
            print("Cannot print 'J' or 'Z'.")
            return
        
        fake_labels = torch.zeros(PRINT_NUM, NUM_CLASSES)
        fake_labels[:, chosen_idx] = 1.

        generated_features = generator.forward(z, fake_labels.to(device))
        imgs = generated_features.reshape(PRINT_NUM, 28, 28)

        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 2.5))

        for i in range(4):
            for j in range(6):
                axes[i][j].imshow(imgs[i*6+j].to(torch.device('cpu')).detach(), cmap='binary')
                axes[i][j].title.set_text(CLASS_LIST[int(fake_labels[i*6+j].argmax())])

        plt.show()


def wasserstein_loss(y_real, y_fake):
    return abs(torch.mean(y_real) - torch.mean(y_fake))