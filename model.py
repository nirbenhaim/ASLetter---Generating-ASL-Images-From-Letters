import torch
import torch.nn as nn

class GAN(torch.nn.Module):
    def __init__(self, latent_dim):
        super(GAN, self).__init__()

        # generator: z [vector] -> image [matrix]
        self.generator_fc_layer = nn.Sequential(

            # nn.Linear(latent_dim, 128),
            # nn.Dropout(p=0.5),
            # nn.LeakyReLU(inplace=True),
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 512),
            # nn.Dropout(p=0.5),
            # nn.LeakyReLU(inplace=True),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4096),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(4096)

        )

        self.generator_conv_layer = nn.Sequential(
            
            # UpConv Layer block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            # UpConv Layer block 2
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # UpConv Layer block 3
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=0),
            
        )


        # discriminator: image [matrix] -> label (0-fake, 1-real)
        self.discriminator_conv_layer = nn.Sequential(
            # nn.Linear(img_size, 128),
            # nn.BatchNorm1d(128), # new
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 1),
            # nn.Sigmoid()

            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=2),
            # `paading=1` is the same as `padding='same'` for 3x3 kernels size
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)

        )

        self.discriminator_fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1)
        )


    def generator_forward(self, z):
        x = self.generator_fc_layer(z)
        x = x.reshape(z.shape[0], 256, 4, 4)
        img = self.generator_conv_layer(x)
        return img
    


    def discriminator_forward(self, img):
        x = self.discriminator_conv_layer(img)
        x = x.view(x.size(0), -1)
        pred = self.discriminator_fc_layer(x)
        return pred.view(-1)
