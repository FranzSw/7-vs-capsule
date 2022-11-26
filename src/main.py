import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file

dataset_path = '/Volumes/X/datasets/NSCLC Radiogenomics'
single_file = '/Volumes/X/datasets/NSCLC Radiogenomics/data/R01-001/10-17-1990-NA-PET CT LUNG CANCER-44295/PETBODYCTAC-72090/1-001.dcm'

def main():
    # data = get_testdata_file(single_file)
    ds = dcmread(single_file)
    arr = ds.pixel_array

    print(ds.file_meta)

    plt.imshow(arr, cmap="gray")
    plt.show()




main()