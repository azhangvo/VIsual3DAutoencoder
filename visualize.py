import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

from models.autoencoder.autoencoder import Autoencoder


def project(mat, phi, omega):
    return np.any(mat, axis=1)


model = Autoencoder()

# test = np.random.rand(1, 32, 32, 32) / 1.99
# test = np.round(test).astype(np.float32)
# test_tensor = torch.from_numpy(test)

test = scipy.io.loadmat("/media/endian/3965-6439/SmallModel40/bench/test/bench_0186.mat")["voxel"]
test = np.pad(test, 1).reshape((1, 32, 32, 32)).astype(np.float32)
test_tensor = torch.from_numpy(test)

print(model.forward(test_tensor))

print(project(test, 0, 0).shape)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(test[0, :], edgecolor='k')

bx = plt.figure().add_subplot()
bx.imshow(project(test, 0, 0)[0, :], cmap="gray", vmin=0, vmax=1)

plt.show()
