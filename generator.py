import torch.nn as nn
from ViT import VisionTransformer


# Generator model
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, ngf, n_residual_layers, vision_transformer=False):
        super(Generator, self).__init__()

        self.vision_transformer = vision_transformer

        self.vit_encoder = VisionTransformer(in_channels=4, patch_size=74, emb_size=256, img_size=(2220, 5820), num_heads=4, num_layers=6)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.residual_blocks = []
        for _ in range(n_residual_layers):
            self.residual_blocks.append(nn.Sequential(
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ngf * 4)
            ))
        self.residual_blocks = nn.Sequential(*self.residual_blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        print(f"entering encoder, x shape: {x.shape}")
        x = self.encoder(x) if not self.vision_transformer else self.vit_encoder(x)
        print(f"entering residual, x shape: {x.shape}")
        x = self.residual_blocks(x)
        print(f"entering decoder, x shape: {x.shape}")
        x = self.decoder(x)
        print(f'GAN done x shape: {x.shape}')
        return x


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_channels, ndf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        print("running discriminator forward")
        return self.model(x)