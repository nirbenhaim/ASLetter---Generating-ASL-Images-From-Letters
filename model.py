import torch
import torch.nn as nn

class GAN(torch.nn.Module):
    def __init__(self, latent_dim, img_size):
        super(GAN, self).__init__()

        # generator: z [vector] -> image [matrix]
        self.generator = nn.Sequential(
            # nn.Linear(latent_dim, 128),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, img_size),
            # nn.Tanh()

            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, img_size),
            nn.BatchNorm1d(img_size),
            nn.LeakyReLU(inplace=True),
        )

        # discriminator: image [matrix] -> label (0-fake, 1-real)
        self.discriminator = nn.Sequential(
            nn.Linear(img_size, 128),
            nn.BatchNorm1d(128), # new
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img
    


    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred.view(-1)


def wasserstein_loss(y_real, y_fake):
    return abs(torch.mean(y_real) - torch.mean(y_fake))