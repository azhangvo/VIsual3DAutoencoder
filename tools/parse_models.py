import argparse
import h5py
import scipy
import glob
import os

parser = argparse.ArgumentParser(description="Process all .off files in given directory and optionally subdirectories "
                                             "into h5 database files")
parser.add_argument("input_dir")
parser.add_argument("out_dir")
parser.add_argument("-r", "--recursive", default=True)

args = parser.parse_args()

input_dir = args.input_dir
out_dir = args.out_dir
recursive = args.recursive


if recursive:
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".mat"):
                continue
            path = os.path.join(root, file)
            mat = scipy.io.loadmat(path)
            print(mat)
else:
    for file in os.listdir(input_dir):
        if not file.endswith(".mat"):
            continue
        path = os.path.join(input_dir, file)
        mat = scipy.io.loadmat(path)
