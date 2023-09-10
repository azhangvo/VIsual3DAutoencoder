import argparse
import os
import re

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")

    args = parser.parse_args()

    input_dir = args.dir

    data_files = {}
    x_train_count = 0
    x_test_count = 0
    for file in os.listdir(input_dir):
        if not re.match(r"^data-\d+\.hdf5", file):
            continue
        path = os.path.join(input_dir, file)
        file = h5py.File(path, "r")
        data_files[path] = {dataset: len(file[dataset]) if dataset in file else 0 for dataset in ["x_train", "x_test"]}
        x_train_count += data_files[path]["x_train"]
        x_test_count += data_files[path]["x_test"]

    layout_x_train = h5py.VirtualLayout(shape=(x_train_count, 32, 32, 32),
                                        dtype=np.int8)
    layout_x_test = h5py.VirtualLayout(shape=(x_test_count, 32, 32, 32),
                                       dtype=np.int8)
    layout_x_train_proj = h5py.VirtualLayout(shape=(x_train_count, 64, 64, 64),
                                             dtype=np.float32)
    layout_x_test_proj = h5py.VirtualLayout(shape=(x_test_count, 64, 64, 64),
                                            dtype=np.float32)

    x_train_index = 0
    x_test_index = 0

    with h5py.File(os.path.join(input_dir, "data.hdf5"), 'w') as f:
        for file, info in data_files.items():
            if info["x_train"]:
                layout_x_train[x_train_index:x_train_index + info["x_train"]] = \
                    h5py.VirtualSource(file, "x_train", shape=(info["x_train"], 32, 32, 32))
                layout_x_train_proj[x_train_index:x_train_index + info["x_train"]] = \
                    h5py.VirtualSource(file, "x_train_proj", shape=(info["x_train"], 64, 64, 64))
                x_train_index += info["x_train"]
            if info["x_test"]:
                layout_x_test[x_test_index:x_test_index + info["x_test"]] = \
                    h5py.VirtualSource(file, "x_test", shape=(info["x_test"], 32, 32, 32))
                layout_x_test_proj[x_test_index:x_test_index + info["x_test"]] = \
                    h5py.VirtualSource(file, "x_test_proj", shape=(info["x_test"], 64, 64, 64))
                x_test_index += info["x_test"]
        f.create_virtual_dataset("x_train", layout_x_train)
        f.create_virtual_dataset("x_train_proj", layout_x_train_proj)
        f.create_virtual_dataset("x_test", layout_x_test)
        f.create_virtual_dataset("x_test_proj", layout_x_test_proj)


if __name__ == "__main__":
    main()
