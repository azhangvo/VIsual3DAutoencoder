import torch.nn


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm3d(num_features=1),
            torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5, 5, 5)),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm3d(num_features=8),
            torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 5, 5)),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm3d(num_features=16),
            torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 5, 5)),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm3d(num_features=32),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5)),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.MaxPool3d(kernel_size=(2, 2, 2)),
            torch.nn.Flatten(),
            # torch.nn.Linear(13824, 11658)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(13824, 16000),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(16000, 32768),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Unflatten(1, (512, 8, 8)),
            torch.nn.ConvTranspose2d(512, 512, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(512, 256, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(256, 256, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(256, 128, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(128, 128, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128, 64, (3, 3), (1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), (1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 64, (3, 3), (1, 1)),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
