import os.path

import h5py
import numpy as np
from torch.utils.data import Dataset


def project(mat, phi, omega):
    return np.any(mat, axis=1)


class Model40Dataset(Dataset):
    def __init__(self, data_dir, database_name):
        self.data_dir = data_dir
        self.data_file = h5py.File(os.path.join(data_dir, "data.hdf5"))
        self.data = self.data_file[database_name]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item].reshape((1, 32, 32, 32)).astype(np.float32), project(
            self.data[item].reshape((32, 32, 32)), 0, 0).repeat(2, axis=0).repeat(2, axis=1).reshape((1, 64, 64)).astype(np.float32)
