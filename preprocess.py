import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Function to ratio of face area to the total image area
def get_ratio_bbox_and_image(full_img_path, bound_box_path):
    img = cv2.imread(full_img_path)
    real_h, real_w, _ = img.shape
    area_image = real_h * real_w
    x1, y1, w1, h1 = get_area_bbox_indices(bound_box_path, real_w, real_h)
    area_bbox = w1 * h1
    return area_bbox / area_image

#function to resize all images to 224*224
def standard_width_height_scaling(real_w, real_h, bbox0, bbox1, bbox2, bbox3):
    x1 = int(int(bbox0) * (float(real_w) / 224))  # bbox[0]
    y1 = int(int(bbox1) * (float(real_h) / 224))  # bbox[1]
    w1 = int(int(bbox2) * (float(real_w) / 224))  # bbox[2]
    h1 = int(int(bbox3) * (float(real_h) / 224))  # bbox[3]
    return x1, y1, w1, h1

#Function to retrieve the coordinates of an image bounding box.
def get_area_bbox_indices(bound_box_path, real_w, real_h):
    bound_box_read = open(bound_box_path, "r")
    bound_box_indices = list()
    for i in bound_box_read:
        bound_box_indices.append(i)
    bbox = bound_box_indices[0].split()
    x1, y1, w1, h1 = standard_width_height_scaling(real_w, real_h,
                                                   bbox[0], bbox[1], bbox[2], bbox[3])
    return x1, y1, w1, h1

#function to crop the image to focus only on the facial region and remove any noise or background
def get_padding_bbox_indices(x1, y1, w1, h1, real_w, real_h, ratio_bbox_and_image):
    x1_padding = x1 - int((w1) * (1 + ratio_bbox_and_image))
    y1_padding = y1 - int((h1) * (1 + ratio_bbox_and_image))
    w1_padding = w1 + int((w1) * (1 + ratio_bbox_and_image))
    h1_padding = h1 + int((h1) * (1 + ratio_bbox_and_image))
    if x1_padding < 0:
        x1_padding = 0
    if y1_padding < 0:
        y1_padding = 0
    if w1_padding > real_w:
        w1_padding = real_w
    if h1_padding > real_h:
        h1_padding = real_h
    return x1_padding, y1_padding, w1_padding, h1_padding


def read_crop_img_with_bbox(full_img_path, bound_box_path):
    img = cv2.imread(full_img_path)
    real_w = img.shape[1]
    real_h = img.shape[0]
    x1, y1, w1, h1 = get_area_bbox_indices(bound_box_path, real_w, real_h)
    return x1, y1, w1, h1, img, real_w, real_h


padding_cropped_storage = []
img_names = []
padding_cropped_labels = []
padding_to_original_map = {}
count_live = 0
count_spoof = 0
dim = (224, 224)
count_limit_live = 50000
count_limit_spoof = 50000

rootdir_train = '/kaggle/input/celeba-spoof-for-face-antispoofing/CelebA_Spoof_/CelebA_Spoof/Data/train'
for file in os.listdir(rootdir_train):
    d = os.path.join(rootdir_train, file)
    if os.path.isdir(d):
        for e in os.listdir(d):
            imgs_path = d + '/' + e + '/'
            for img_path in os.listdir(imgs_path):
                if (img_path.endswith(".jpg")):
                    full_img_path = imgs_path + img_path
                    bound_box_path = full_img_path[0:-4] + '_BB.txt'
                    x1, y1, w1, h1, img, real_w, real_h = read_crop_img_with_bbox(full_img_path, bound_box_path)
                    ratio_bbox_and_image = get_ratio_bbox_and_image(full_img_path, bound_box_path)
                    x1_padding, y1_padding, w1_padding, h1_padding = get_padding_bbox_indices(x1, y1, w1, h1,
                                                                                              real_w, real_h,
                                                                                              ratio_bbox_and_image)
                    padding_img = img[y1_padding:y1 + h1_padding, x1_padding:x1 + w1_padding]
                    try:
                        if (e == 'live' and count_live >= count_limit_live) or (
                                e == 'spoof' and count_spoof >= count_limit_spoof):
                            continue
                        resized_padding_img = cv2.resize(padding_img, dim, interpolation=cv2.INTER_AREA)
                        padding_cropped_storage.append(resized_padding_img)
                        padding_to_original_map[len(padding_cropped_storage) - 1] = full_img_path
                        if e == 'live':
                            count_live = count_live + 1
                            padding_cropped_labels.append(1)
                        elif e == 'spoof':
                            count_spoof = count_spoof + 1
                            padding_cropped_labels.append(0)
                    except:
                        continue

                    img_names.append(img_path)

                    if (count_live == count_limit_live and e == 'live') or (
                            count_spoof == count_limit_spoof and e == 'spoof'):
                        break
            if count_live >= count_limit_live and count_spoof >= count_limit_spoof:
                break
    if count_live >= count_limit_live and count_spoof >= count_limit_spoof:
        print("DONE Extracting ")
        break

X = np.asarray(padding_cropped_storage)
y = np.asarray(padding_cropped_labels)
np.savez('anti_spoofing_data.npz', X, y)
print("DONE SAVING DATA WITH NPZ")

