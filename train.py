import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from data import CustomDataset
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
from model import GAN
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

CLASS_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def train_model(data_path):
    # setting
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    random_seed = 123
    generator_learning_rate = 0.001
    discriminator_learning_rate = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LATENT_DIM = 100 # latent vectors dimension [z]
    IMG_SHAPE = (1, 28, 28) # MNIST has 1 color channel, each image 28x8 pixels
    IMG_SIZE = 1
    for x in IMG_SHAPE:
        IMG_SIZE *= x


    file_path_train = os.path.join(data_path, 'sign_mnist_train.csv')
    file_path_test = os.path.join(data_path, 'sign_mnist_test.csv')

    #dataset
    # to 0-1 range
    train_dataset = CustomDataset(csv_file=file_path_train)
    
    test_dataset = CustomDataset(csv_file=file_path_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # # Checking the dataset
    # for images, labels in train_loader:
    #     print('Image batch dimensions:', images.shape)
    #     print('Image label dimensions:', labels.shape)
    #     break

    # # let's see some digits
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print("shape: \n", example_data.shape)
    # fig = plt.figure()
    # for i in range(6):
    #     ax = fig.add_subplot(2,3,i+1)
    #     ax.imshow(example_data[i], cmap='gray', interpolation='none')
    #     ax.set_title("Ground Truth: {}".format(CLASS_LIST[int(example_targets[i])]))
    #     ax.set_axis_off()
    # plt.tight_layout()

    # plt.show()

    # constant the seed
    torch.manual_seed(random_seed)

    # build the model, send it ti the device
    model = GAN(LATENT_DIM, IMG_SIZE).to(device)

    # optimizers: we have one for the generator and one for the discriminator
    # that way, we can update only one of the modules, while the other one is "frozen"
    optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
    optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)

    # training
    start_time = time.time()
    discr_costs = []
    gener_costs = []

    for epoch in range(NUM_EPOCHS):
        model = model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = (features - 0.5) * 2.0 # normalize between [-1, 1]
            features = features.view(-1, IMG_SIZE).to(device)
            targets = targets.to(device)

            # generate fake and real labels
            valid = torch.ones(targets.size(0)).float().to(device)
            fake = torch.zeros(targets.size(0)).float().to(device)

            ### FORWARD PASS AND BACKPROPAGATION

            # --------------------------
            # Train Generator
            # --------------------------

            # Make new images
            z = torch.zeros((targets.size(0), LATENT_DIM)).uniform_(-1.0, 1.0).to(device) # can also be N(0,1)
            generated_features = model.generator_forward(z)

            # Loss for fooling the discriminator
            discr_pred = model.discriminator_forward(generated_features)

            # here we use the `valid` labels because we want the discriminator to "think"
            # the generated samples are real
            gener_loss = F.binary_cross_entropy(discr_pred, valid)

            optim_gener.zero_grad()
            gener_loss.backward()
            optim_gener.step()

            # --------------------------
            # Train Discriminator
            # --------------------------

            discr_pred_real = model.discriminator_forward(features.view(-1, IMG_SIZE))
            real_loss = F.binary_cross_entropy(discr_pred_real, valid)

            # here we use the `fake` labels when training the discriminator
            discr_pred_fake = model.discriminator_forward(generated_features.detach())
            fake_loss = F.binary_cross_entropy(discr_pred_fake, fake)

            discr_loss = 0.5 * (real_loss + fake_loss)
            optim_discr.zero_grad()
            discr_loss.backward()
            optim_discr.step()

            discr_costs.append(discr_loss)
            gener_costs.append(gener_loss)


            ### LOGGING
            if not batch_idx % 100:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f'%(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), gener_loss, discr_loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # save model
    PATH = "model.pth"
    torch.save(model.state_dict(), PATH)

    # Evaluation

    for idx in range(len(gener_costs)):
        gener_costs[idx] = gener_costs[idx].cpu().detach().numpy()
    for idx in range(len(discr_costs)):
        discr_costs[idx] = discr_costs[idx].cpu().detach().numpy()

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(gener_costs)), gener_costs, label='Generator loss')
    ax1.plot(range(len(discr_costs)), discr_costs, label='Discriminator loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(NUM_EPOCHS+1))
    iter_per_epoch = len(train_loader)
    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticklabels(newlabel[::10])
    ax2.set_xticks(newpos[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.show()