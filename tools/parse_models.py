import argparse
import h5py
import numpy as np
import scipy
import glob
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process all .off files in given directory and optionally subdirectories "
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

data_file = h5py.File(data_file_path, "w")

data_file.create_dataset("x_train", (len(x_train_paths), 32, 32, 32), np.int8)
data_file.create_dataset("x_test", (len(x_test_paths), 32, 32, 32), np.int8)

with tqdm(total=num_files) as pbar:
    for i, path in enumerate(x_train_paths):
        mat = scipy.io.loadmat(path)
        voxels = mat['voxel']
        voxels = np.pad(voxels, 1)
        data_file["x_train"][i] = voxels
        pbar.update(1)
    for i, path in enumerate(x_test_paths):
        mat = scipy.io.loadmat(path)
        voxels = mat['voxel']
        voxels = np.pad(voxels, 1)
        data_file["x_test"][i] = voxels
        pbar.update(1)
