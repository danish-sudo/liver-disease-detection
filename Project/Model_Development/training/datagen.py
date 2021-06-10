import numpy as np

import pydicom
import cv2
from PIL import Image
from tensorflow import keras
import os
import matplotlib.pyplot as plt

image_size = 256
image_datatype = float

masks_path = '../../processed_data/masks'
training_path = '../../processed_data/patients'

dataset_path = "../../processed_data"


def split_data():
    ct_scans = []
    for file in os.listdir(os.path.join(dataset_path, "patients")):
        ct_scans.append(file)
    print("Total CT scans :" + str(len(ct_scans)))

    valid_data_size = len(ct_scans) // 5

    valid_set = ct_scans[:valid_data_size]
    train_set = ct_scans[valid_data_size:]
    return train_set, valid_set


def normalize_scans(img):
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def preprocess_scans(img_scan):
    img_scan[img_scan > 1200] = 0
    img_scan = np.clip(img_scan, -100, 400)
    img_scan = normalize_scans(img_scan)
    img_scan = img_scan * 255
    img_scan = img_scan.astype('uint8')
    img_scan = cv2.equalizeHist(img_scan)
    img_scan = normalize_scans(img_scan)
    return img_scan


class LiverDataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, scan_size=256):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = scan_size
        self.on_epoch_end()

    def __load__(self, id_name):
        patient_id = id_name.split('_')
        image_path = os.path.join(self.path, "patients", id_name)
        mask_path = os.path.join(self.path, "masks")
        dicom_image = pydicom.dcmread(image_path)
        image = preprocess_scans(dicom_image.pixel_array)
        image = normalize_scans(image)
        image = np.array(Image.fromarray(image).resize([image_size, image_size])).astype(image_datatype)
        mask = pydicom.dcmread(os.path.join(mask_path, patient_id[0] + '_liver', id_name)).pixel_array
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)
        mask = np.array(Image.fromarray(mask).resize([image_size, image_size])).astype(image_datatype)
        mask = mask[:, :, np.newaxis]
        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            _img = np.stack((_img,) * 3, axis=-1)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, scan_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = scan_size
        self.on_epoch_end()

    def __load__(self, id_name):

        tumor_volume = None

        image_path = os.path.join(self.path, "patients", id_name)
        mask_path = os.path.join(self.path, "masks")
        all_masks = os.listdir(mask_path)
        dicom_image = pydicom.dcmread(image_path)

        image = preprocess_scans(dicom_image.pixel_array)

        liver_mask_id = id_name.split('_')
        liver_mask = pydicom.dcmread(os.path.join(mask_path, liver_mask_id[0] + '_liver', id_name)).pixel_array

        image = np.multiply(image, np.clip(liver_mask, 0, 1))

        image = np.array(Image.fromarray(image).resize([image_size, image_size])).astype(image_datatype)
        mask = cv2.imread(os.path.join(masks_path, 'merged_livertumors', id_name + '.jpg'))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        retval, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        mask = np.array(Image.fromarray(mask).resize([image_size, image_size])).astype(image_datatype)
        mask = mask // 255

        mask = mask[:, :, np.newaxis]

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]
        image = []
        mask = []
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            _img = np.stack((_img,) * 3, axis=-1)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def display_scan(*args, **kwargs):
    cmap = kwargs.get('cmap', 'gray')
    title = kwargs.get('title', '')
    if len(args) == 0:
        raise ValueError("NO Image to show")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap] * n
        if type(title) == str:
            title = [title] * n
        plt.figure(figsize=(n * 5, 10))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()
