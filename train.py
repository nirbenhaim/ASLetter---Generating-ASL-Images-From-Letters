import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
import os
import time
from data import CustomDataset
from model import Generator, Discriminator
from util import wasserstein_loss
import matplotlib.pyplot as plt

CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def train_model(data_path, model_path):

    BATCH_SIZE = 64
    LATENT_DIM = 100
    NUM_CLASSES = 26
    NUM_EPOCHS = 200
    lr_gen = 5e-3
    lr_disc = 2e-3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_path_train = os.path.join(data_path, 'sign_mnist_train.csv')
    file_path_test = os.path.join(data_path, 'sign_mnist_test.csv')

    train_dataset = CustomDataset(csv_file=file_path_train)
    
    test_dataset = CustomDataset(csv_file=file_path_test) 

    all_data = train_dataset

    for test_data in test_dataset.data:
        all_data.append(test_data)

    all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM, NUM_CLASSES).to(device)
    discriminator = Discriminator(NUM_CLASSES).to(device)
    
    # Loss function
    adversarial_loss = wasserstein_loss
    
    # Optimizers
    optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_gen)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr_disc)
    scheduler_gen = ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_disc = ExponentialLR(optimizer_D, gamma=0.95)

    torch.manual_seed(123)
    start_time = time.time()
    discr_costs = []
    gener_costs = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for i, (imgs, labels) in enumerate(all_data_loader):
            batch_size = imgs.shape[0]

            imgs = (imgs - 0.5) * 2.0
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Adversarial ground truths
            valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(device)

            # Configure input
            real_imgs = imgs
            labels = labels.view(labels.size(0), -1)
            labels_reshape = torch.empty((batch_size, NUM_CLASSES))
            for idx, label in enumerate(labels):
                labels_temp = torch.zeros(1, NUM_CLASSES).squeeze()
                labels_temp[int(label)] = 1.
                labels_reshape[idx] = labels_temp

            labels_reshape = labels_reshape.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Discriminator loss on real data
            real_validity = discriminator(real_imgs, labels_reshape)
            d_real_loss = adversarial_loss(real_validity, valid)

            # Discriminator loss on fake data
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_labels = torch.LongTensor(batch_size).random_(0, NUM_CLASSES-2)
            fake_labels_reshape = torch.empty((batch_size, NUM_CLASSES))
            for idx, fake_label in enumerate(fake_labels):
                fake_labels_temp = torch.zeros(1, NUM_CLASSES).squeeze()
                if fake_label >= 9:
                    fake_label += 1
                fake_labels_temp[fake_label] = 1.
                fake_labels_reshape[idx] = fake_labels_temp

            fake_labels_reshape = fake_labels_reshape.to(device)
            
            fake_imgs = generator(noise, fake_labels_reshape)
            fake_validity = discriminator(fake_imgs.detach(), fake_labels_reshape)
            d_fake_loss = adversarial_loss(fake_validity, fake)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generator loss
            fake_validity = discriminator(fake_imgs, fake_labels_reshape)
            g_loss = adversarial_loss(fake_validity, valid)
            g_loss.backward()
            optimizer_G.step()
            
            discr_costs.append(d_loss.item())
            gener_costs.append(g_loss.item())

            if (i+1) % 100 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i+1}/{len(all_data_loader)}] "
                    f"[D loss: {discr_costs[-1]:.4f}] [G loss: {gener_costs[-1]:.4f}]")
            
        scheduler_gen.step()
        scheduler_disc.step()
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # save model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    gen_save_path = os.path.join(model_path, 'generator.pth')
    dis_save_path = os.path.join(model_path, 'discriminator.pth')
    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), dis_save_path)

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(gener_costs)), gener_costs, label='Generator loss')
    ax1.plot(range(len(discr_costs)), discr_costs, label='Discriminator loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(NUM_EPOCHS+1))
    iter_per_epoch = len(all_data_loader)
    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticklabels(newlabel[::10])
    ax2.set_xticks(newpos[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.show()
