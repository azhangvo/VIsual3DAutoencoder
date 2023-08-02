import argparse
import subprocess
import time

import numpy as np

import binvox_rw
from scipy.io import savemat
import os
import shutil

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert .off files into voxel files")
parser.add_argument("input_dir")
parser.add_argument("-r", "--recursive", action="store_true")
parser.add_argument("-f", "--flush", action="store_true")

args = parser.parse_args()

input_dir = args.input_dir
recursive = args.recursive
flush = args.flush

if flush:
    inp = input("Are you sure you would like to flush .binvox and .mat files? (y/n): ")

    if inp != "y" and inp != "Y":
        print("Aborting")
        exit(0)

    if recursive:
        num_files = sum(len(files) for _, _, files in os.walk(input_dir))
        with tqdm(total=num_files) as pbar:
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".binvox"):
                        pbar.update(1)
                        path = os.path.join(root, file)
                        os.remove(path)
                    else:
                        num_files -= 1
                        pbar.total = num_files
    else:
        num_files = len(os.listdir(input_dir))
        with tqdm(total=num_files) as pbar:
            for file in os.listdir(input_dir):
                if file.endswith(".binvox"):
                    pbar.update(1)
                    path = os.path.join(input_dir, file)
                    os.remove(path)
                else:
                    num_files -= 1
                    pbar.total = num_files

if shutil.which("binvox") is None:
    raise RuntimeError("binvox is required for this script")

if flush:
    print("\nFlushing completed\n")
    time.sleep(1)
    for i in range(1, 4):
        print("Starting" + "." * i, end="\r")
        time.sleep(1)

if recursive:
    num_files = sum(len(files) for _, _, files in os.walk(input_dir))
    with tqdm(total=num_files) as pbar:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".off"):
                    num_files -= 1
                    pbar.total = num_files
                    continue
                pbar.update(1)
                path = os.path.join(root, file)
                pbar.set_description(f"Processing {path}")
                mat_path = os.path.splitext(path)[0] + ".mat"
                if not os.path.exists(mat_path):
                    binvox_path = os.path.splitext(path)[0] + ".binvox"
                    if not os.path.exists(binvox_path):
                        subprocess.run(["binvox", "-d", "30", path], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                    model = binvox_rw.read_as_3d_array(open(binvox_path, 'rb'))
                    voxels = model.data.astype(np.int32).transpose(0, 2, 1)
                    voxels[:, 1, :] = -voxels[:, 1, :]
                    savemat(mat_path, {'voxel': voxels})
else:
    num_files = len(os.listdir(input_dir))
    with tqdm(total=num_files) as pbar:
        for file in os.listdir(input_dir):
            if not file.endswith(".off"):
                num_files -= 1
                pbar.total = num_files
                continue
            pbar.update(1)
            path = os.path.join(root, file)
            pbar.set_description(f"Processing {path}")
            mat_path = os.path.splitext(path)[0] + ".mat"
            if not os.path.exists(mat_path):
                binvox_path = os.path.splitext(path)[0] + ".binvox"
                if not os.path.exists(binvox_path):
                    subprocess.run(["binvox", "-d", "30", path], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                model = binvox_rw.read_as_3d_array(open(binvox_path, 'rb'))
                voxels = model.data.astype(np.int32).transpose(0, 2, 1)
                voxels[:, 1, :] = -voxels[:, 1, :]
                savemat(mat_path, {'voxel': voxels})
