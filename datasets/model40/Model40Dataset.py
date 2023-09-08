import math
import os.path
import random

import h5py
import numpy as np
from numba import jit
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


@jit(nopython=True)
def projectHelper(base, coords, offset):
    for coord in coords:
        x, y, z = coord
        x = 64 - round(x + offset / 2)
        z = round(z + offset / 2)
        if 0 <= x < 64 and 0 <= z < 64 and (np.isnan(base[x, z]) or y < base[x, z]):
            base[x, z] = y


# @jit(nopython=True, fastmath=False, parallel=True)
def project(inp, phi, omega):
    r = 60.0
    mat = inp.copy()
    mat = mat.repeat(4, axis=0).repeat(4, axis=1).repeat(4, axis=2)
    size = mat.shape[0]
    offset = (size - 1) / 2.0
    # X Y Z - Y is vertical
    #   |
    #  / \
    # x   z
    camera_pos = np.asarray(
        [[r * math.sin(omega) * math.cos(phi)], [r * math.cos(omega)], [r * math.sin(omega) * math.sin(phi)]])

    coords = np.asarray(mat.nonzero()) - offset
    coords /= 3

    coords = coords - camera_pos

    # rotation_matrix = R.from_euler('yz', (phi, omega - np.pi))
    # coords = rotation_matrix.apply(coords)

    rotation_matrix = np.matmul(np.asarray([[np.cos(omega - np.pi), -np.sin(omega - np.pi), 0],
                                            [np.sin(omega - np.pi), np.cos(omega - np.pi), 0],
                                            [0, 0, 1]]),
                                np.asarray([[np.cos(phi), 0, np.sin(phi)],
                                            [0, 1, 0],
                                            [-np.sin(phi), 0, np.cos(phi)]]))

    coords = np.matmul(rotation_matrix, coords).transpose()
    # coords = np.matmul(np.asarray([[np.cos(phi), 0, np.sin(phi)],
    #                                [0, 1, 0],
    #                                [-np.sin(phi), 0, np.cos(phi)]]), coords)
    # coords = np.matmul(np.asarray([[np.cos(omega - np.pi), -np.sin(omega - np.pi), 0],
    #                                [np.sin(omega - np.pi), np.cos(omega - np.pi), 0],
    #                                [0, 0, 1]]), coords)
    # coords = coords.transpose()

    base = np.empty((64, 64), np.float32)
    base[:] = np.nan
    projectHelper(base, coords, offset)

    # denom = np.nanmax(base) - np.nanmin(base) if np.nanmax(base) - np.nanmin(base) != 0 else \
    #     np.nanmax(base)
    # base = np.nan_to_num((base > 0) * 0.3 + (base - np.nanmin(base)) / denom * 0.7)

    # base = (base - np.nanmin(base)) / (np.nanmax(base) - np.nanmin(base)) * 0.7 + 0.3 if np.nanmax(base) != np.nanmin(
    #     base) else base / np.nanmax(base)
    base = (-base + np.nanmax(base)) / (np.nanmax(base) - np.nanmin(base)) * 0.7 + 0.3 if np.nanmax(base) != np.nanmin(
        base) else base / np.nanmax(base)
    base = np.nan_to_num(base)

    return base

    # return np.any(mat, axis=1)


class Model40Dataset(Dataset):
    def __init__(self, data_dir, database_name):
        self.data_dir = data_dir
        self.data_file = h5py.File(os.path.join(data_dir, "data.hdf5"))
        self.data = self.data_file[database_name]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        phi = random.random() * 2 * np.pi
        omega = random.random() * np.pi
        return (self.data[item].reshape((1, 32, 32, 32)).astype(np.float32),
                np.array([phi, omega, np.sin(phi), np.cos(phi), np.sin(omega), np.cos(omega)], dtype=np.float32)), \
               project(self.data[item].reshape((32, 32, 32)), phi, omega) \
                   .reshape(
                   (1, 64, 64)).astype(np.float32)
        # .repeat(2, axis=0) \
        # .repeat(2, axis=1)
