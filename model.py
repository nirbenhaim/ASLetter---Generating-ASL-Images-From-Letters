import torch
import torch.nn as nn

# # Generator model
# class Generator(nn.Module):
#     def __init__(self, latent_dim, num_classes):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.num_classes = num_classes
        
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim + num_classes, 256),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024, 28 * 28),
#             nn.Tanh()
#         )
    
#     def forward(self, noise, labels):
#         gen_input = torch.cat((noise, labels), dim=1)
#         img = self.model(gen_input)
#         img = img.view(img.size(0), 1, 28, 28)
#         return img

# # Discriminator model
# class Discriminator(nn.Module):
#     def __init__(self, num_classes):
#         super(Discriminator, self).__init__()
#         self.num_classes = num_classes
        
#         self.model = nn.Sequential(
#             nn.Linear(28 * 28 + num_classes, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, img, labels):
#         img_flat = img.view(img.size(0), -1)
#         disc_input = torch.cat((img_flat, labels), dim=1)
#         validity = self.model(disc_input)
#         return validity
    

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        # generator: z [vector] -> image [matrix]
        self.fc_layer = nn.Sequential(

            nn.Linear(latent_dim + num_classes, 1024),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4096),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(4096)

        )

        self.conv_layer = nn.Sequential(
            
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

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), dim=1)
        z = self.fc_layer(gen_input)
        x = z.reshape(z.shape[0], 256, 4, 4)
        img = self.conv_layer(x)
        return img


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(Discriminator, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=(1+num_classes), out_channels=32, kernel_size=3, padding=2),
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

        self.fc_layer = nn.Sequential(
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
    
    def forward(self, img, labels):
        labels = labels.view(labels.size(0), self.num_classes, 1, 1)
        labels = labels.repeat(1, 1, img.size(2), img.size(3))
        # img_flat = img.view(img.size(0), -1)
        disc_input = torch.cat((img, labels), dim=1)
        conv_output = self.conv_layer(disc_input)
        x = conv_output.view(conv_output.size(0), -1)
        validity = self.fc_layer(x)
        return validity