import random

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.model40.Model40Dataset import Model40Dataset
from models.autoencoder.autoencoder import Autoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = DataLoader(Model40Dataset("/media/endian/3965-6439/SmallModel40/", "x_test"), shuffle=True)
iterable = iter(dataset)


def project(mat, phi, omega):
    return np.any(mat, axis=1)


def visualize_model(model):
    with torch.no_grad():
        inputs, labels = next(iterable)
        inputs, angles = inputs
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs, angles)

        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.voxels(inputs.numpy()[0, 0, :].swapaxes(1, 2), edgecolor='k')
        # plt.show()

        bx = plt.figure().add_subplot()
        bx.imshow(labels.numpy()[0, 0, :], cmap="gray", vmin=0, vmax=1)
        # plt.show()

        cx = plt.figure().add_subplot()
        cx.imshow(outputs.numpy()[0, 0, :], cmap="gray", vmin=0, vmax=1)
        plt.show()


if __name__ == "__main__":
    model = Autoencoder()

    model.load_state_dict(torch.load("./best_model_params.pt"))

    visualize_model(model)

# test = np.random.rand(1, 32, 32, 32) / 1.99
# test = np.round(test).astype(np.float32)
# test_tensor = torch.from_numpy(test)

# test = scipy.io.loadmat("/media/endian/3965-6439/SmallModel40/bench/test/bench_0186.mat")["voxel"]
# test = np.pad(test, 1).reshape((1, 32, 32, 32)).astype(np.float32)
# test_tensor = torch.from_numpy(test)
#
# print(model.forward(test_tensor))
#
# print(project(test, 0, 0).shape)
#
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(test[0, :], edgecolor='k')
#
# bx = plt.figure().add_subplot()
# bx.imshow(project(test, 0, 0)[0, :], cmap="gray", vmin=0, vmax=1)
#
# plt.show()
