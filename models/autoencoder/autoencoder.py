import torch.nn


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, (5, 5, 5)),
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Conv3d(8, 40, (5, 5, 5)),
            torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Flatten(0),
            torch.nn.Linear(5000, 2000)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2000, 10240),
            torch.nn.Unflatten(0, (10, 32, 32)),
            torch.nn.Conv2d(10, 4, (3, 3), (1, 1), 1),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(4, 1, (3, 3), (1, 1), 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
