import os
import sys
import time
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.model40.Model40Dataset import Model40Dataset
from models.autoencoder.autoencoder import Autoencoder

dataloaders = {"train": DataLoader(Model40Dataset("/media/endian/3965-6439/SmallModel40/", "x_train"), batch_size=4,
                                   shuffle=True, num_workers=8),
               "val": DataLoader(Model40Dataset("/media/endian/3965-6439/SmallModel40/", "x_test"), batch_size=4,
                                 shuffle=True, num_workers=8)}
dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 1e9

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0
                running_corrects = 0

                # Iterate over data.
                pbar = tqdm(dataloaders[phase], file=sys.stdout)
                pbar.set_description(f"{phase} Loss: ---")
                for inputs, labels in pbar:
                    inputs, angles = inputs
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, angles)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.set_description(f"{phase} Loss: {running_loss / (pbar.n + 1):.4f}")
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print()

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    model = Autoencoder()

    # model.load_state_dict(torch.load("./best_model_params.pt"))

    criterion = torch.nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, 20)

    torch.save(model.state_dict(), "./best_model_params.pt")
