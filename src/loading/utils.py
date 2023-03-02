import os

# import cv2
import numpy as np
import pydicom as dicomio
import SimpleITK as sitk
from PIL import Image


def loadlist(path):
    list = os.listdir(path)
    list.sort()
    return list


def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    if len(img_array.shape) == 3:
        frame_num, width, height = img_array.shape
        ch = 1
        return img_array, frame_num, width, height, ch
    elif len(img_array.shape) == 4:
        frame_num, width, height, ch = img_array.shape
        return img_array, frame_num, width, height, ch


def loadFileInformation(filename):
    information = {}
    ds = dicomio.read_file(filename, force=True)
    information["dicom_num"] = ds.SOPInstanceUID
    # information['PatientID'] = ds.PatientID
    # information['PatientName'] = ds.PatientName
    # information['PatientBirthDate'] = ds.PatientBirthDate
    # information['PatientSex'] = ds.PatientSex
    # information['StudyID'] = ds.StudyID
    # information['StudyDate'] = ds.StudyDate
    # information['StudyTime'] = ds.StudyTime
    # information['InstitutionName'] = ds.InstitutionName
    # information['Manufacturer'] = ds.Manufacturer
    # information['NumberOfFrames'] = ds.NumberOfFrames
    return information


def showImage(img_array):
    img_array = img_array
    img_bitmap = Image.fromarray(img_array)
    # img_bitmap.show()
    return img_bitmap


def MatrixToImage(data, ch):
    # data = (data+1024)*0.125
    # new_im = Image.fromarray(data.astype(np.uint8))
    # new_im.show()
    if ch == 3:
        pass
        # img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    if ch == 1:
        data = (data + 1024) * 0.125
        img_rgb = data.astype(np.uint8)
    return img_rgb


def PETToImage(img_array, color_reversed=True):
    info = np.finfo(img_array.dtype)
    data = img_array.astype(np.float64) / np.max(img_array)
    if color_reversed is True:
        data = 255 - 255 * data
    elif color_reversed is False:
        data = 255 * data
    # data = (data + 1024) * 0.125
    img = data.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    # cv2.imshow('test', img)
    return img


def dfs_showdir(path, depth):
    if depth == 0:
        print("root:[" + path + "]")

    for item in os.listdir(path):
        if ".git" not in item:
            print("|      " * depth + "+--" + item)

            newitem = path + "/" + item
            if os.path.isdir(newitem):
                dfs_showdir(newitem, depth + 1)
    # print(path_list)


def isdir(x):
    return os.path.isdir(x) and x != ".svn"


def mkfloders(src, tar):
    paths = os.listdir(src)
    paths = map(lambda name: os.path.join(src, name), paths)
    paths = list(filter(isdir, paths))
    if len(paths) <= 0:
        return
    for i in paths:
        (filepath, filename) = os.path.split(i)
        targetpath = os.path.join(tar, filename)
        not os.path.isdir(targetpath) and os.mkdir(targetpath)
        mkfloders(i, targetpath)


def mkdir(path):

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + " successfully be made!")
        return True
    else:
        print(path + " the folder already existed!")
        return False
