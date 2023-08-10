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
            torch.nn.Linear(13824, 11658)
        )
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(8006, 8192),
            torch.nn.Unflatten(1, (16, 27, 27)),
            torch.nn.ConvTranspose2d(16, 8, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(8, 4, (3, 3)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(4, 1, (3, 3), (1, 1)),
            torch.nn.LeakyReLU(0.2),
            # torch.nn.Sigmoid()
        )

    def forward(self, x, angles):
        encoded = self.encoder(x)
        decoded = self.decoder(torch.cat((encoded, angles), 1))
        return decoded
