import argparse
import math
import multiprocessing
from itertools import chain

import h5py
import numpy as np
import scipy
import glob
import os

from numba import jit
from tqdm import tqdm


@jit(nopython=True)
def projectHelper(base, coords, offset):
    for coord in coords:
        x, y, z = coord
        x = 64 - round(x + offset / 2)
        z = round(z + offset / 2)
        if 0 <= x < 64 and 0 <= z < 64 and (np.isnan(base[x, z]) or y < base[x, z]):
            base[x, z] = y


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


def process_chunk(data_file_path, paths, completed):
    data_file = h5py.File(data_file_path, "w")

    count = {}

    for database, _ in paths:
        if database in count.keys():
            count[database] = count[database] + 1
        else:
            count[database] = 1
    for database, num in count.items():
        if database in ["x_train", "x_test"]:
            data_file.create_dataset(database, (num, 32, 32, 32), np.int8)
            data_file.create_dataset(f"{database}_proj", (num, 64, 64, 64), np.float32)
        count[database] = 0

    for database, path in paths:
        mat = scipy.io.loadmat(path)
        voxels = mat['voxel']
        voxels = np.pad(voxels, 1)
        data_file[database][count[database]] = voxels
        for j in range(8):
            for k in range(8):
                data_file[f"{database}_proj"][count[database], j * 8 + k] = project(voxels.reshape((32, 32, 32)), np.pi / 4 * j,
                                                                      np.pi / 9 * (k + 1))
        completed.value += 1
        count[database] = count[database] + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all .off files in given directory and optionally subdirectories "
                    "into h5 database files")
    parser.add_argument("input_dir")
    parser.add_argument("out_dir")
    parser.add_argument("-r", "--recursive", action="store_true")

    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    recursive = args.recursive


    def is_for_training(file_path: str):
        return "train" in file_path


    def is_for_testing(file_path: str):
        return "test" in file_path


    num_files = 0
    x_train_paths = []
    x_test_paths = []
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".mat"):
                    continue
                path = os.path.join(root, file)
                if is_for_training(path):
                    x_train_paths.append(path)
                    num_files += 1
                elif is_for_testing(path):
                    x_test_paths.append(path)
                    num_files += 1
    else:
        for file in os.listdir(input_dir):
            if not file.endswith(".mat"):
                continue
            path = os.path.join(input_dir, file)
            if is_for_training(path):
                x_train_paths.append(path)
                num_files += 1
            elif is_for_testing(path):
                x_test_paths.append(path)
                num_files += 1

    train_split_path = os.path.join(out_dir, "train.csv")
    test_split_path = os.path.join(out_dir, "test.csv")
    data_file_path = os.path.join(out_dir, "data.hdf5")

    existing_files = []
    for path in [train_split_path, test_split_path, data_file_path]:
        if os.path.exists(path):
            existing_files.append(path)

    if len(existing_files) > 0:
        inp = input(", ".join(existing_files) + " already exist. Overwrite? (y/n): ")
        if inp != "y" and inp != "Y":
            print("Aborting")
            exit(0)
        for path in existing_files:
            os.remove(path)

    np.savetxt(train_split_path, x_train_paths, delimiter=",", fmt="%s")
    np.savetxt(test_split_path, x_test_paths, delimiter=",", fmt="%s")

    # data_file = h5py.File(data_file_path, "w")

    # data_file.create_dataset("x_train", (len(x_train_paths), 32, 32, 32), np.int8)
    # data_file.create_dataset("x_train_proj", (len(x_train_paths), 64, 64, 64), np.float32)
    # data_file.create_dataset("x_test", (len(x_test_paths), 32, 32, 32), np.int8)
    # data_file.create_dataset("x_test_proj", (len(x_train_paths), 64, 64, 64), np.float32)

    processors = 8

    combined_paths = list(chain(zip(["x_train"] * len(x_train_paths), x_train_paths),
                                zip(["x_test"] * len(x_test_paths), x_test_paths)))

    chunks = [
        combined_paths[
        i * math.ceil(len(combined_paths) / processors):
        i * math.ceil(len(combined_paths) / processors) + math.ceil(len(combined_paths) / processors)]
        for i in range(processors)
    ]

    with tqdm(total=num_files) as pbar:
        completed = multiprocessing.Value('d', 0)

        processes = []
        for i in range(processors):
            file_path = os.path.join(out_dir, f"data-{i}.hdf5")
            p = multiprocessing.Process(target=process_chunk, args=(file_path, chunks[i], completed))
            processes.append(p)
            p.start()

        last = 0
        while not all(not process.is_alive() for process in processes):
            v = completed.value
            pbar.update(v - last)
            last = v



    # with tqdm(total=num_files) as pbar:
    #     for i, path in enumerate(x_train_paths):
    #         mat = scipy.io.loadmat(path)
    #         voxels = mat['voxel']
    #         voxels = np.pad(voxels, 1)
    #         data_file["x_train"][i] = voxels
    #         for j in range(8):
    #             for k in range(8):
    #                 data_file["x_train_proj"][i, j * 8 + k] = project(voxels.reshape((32, 32, 32)), np.pi / 4 * j,
    #                                                                   np.pi / 9 * (k + 1))
    #         pbar.update(1)
    #     for i, path in enumerate(x_test_paths):
    #         mat = scipy.io.loadmat(path)
    #         voxels = mat['voxel']
    #         voxels = np.pad(voxels, 1)
    #         data_file["x_test"][i] = voxels
    #         for j in range(8):
    #             for k in range(8):
    #                 data_file["x_test_proj"][i, j * 8 + k] = project(voxels.reshape((32, 32, 32)), np.pi / 4 * j,
    #                                                                  np.pi / 9 * (k + 1))
    #         pbar.update(1)
