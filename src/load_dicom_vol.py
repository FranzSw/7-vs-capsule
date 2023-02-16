import pydicom
import numpy as np
import os

def load_volume(dir_path: str):
    # load the DICOM files
    files = [pydicom.dcmread(os.path.join(dir_path, fname)) for fname in os.listdir(dir_path)]

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d


    spacing = np.concatenate((ps, [ss]))

    npVol = img3d
    print("Max value",np.max(img3d), "min value", np.min(img3d), "mean", np.mean(img3d),"std", np.std(img3d))
    return npVol