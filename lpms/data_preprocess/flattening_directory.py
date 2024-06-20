import preprocess_module as lib
import os
from torchvision import datasets, transforms
import sys

src_folder = "image"
dst_folder = "data"
if not os.path.exists(dst_folder):
    print(f"flattening {dst_folder}")
    lib.copy_folder(src_folder=src_folder, dst_folder=dst_folder)
    lib.flatten_subfolders(dst_folder)