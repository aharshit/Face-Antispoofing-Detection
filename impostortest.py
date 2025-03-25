import cv2
import numpy as np
import os

dim = (224, 224)
# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_padding_bbox_indices(x1, y1, w1, h1, real_w, real_h, ratio_bbox_and_image):

    x1_padding = x1 - int((w1) * (1 + ratio_bbox_and_image))
    y1_padding = y1 - int((h1) * (1 + ratio_bbox_and_image))
    w1_padding = w1 + int((w1) * (1 + ratio_bbox_and_image))
    h1_padding = h1 + int((h1) * (1 + ratio_bbox_and_image))

    # Ensure the padding does not go out of the image boundaries
    if x1_padding < 0:
        x1_padding = 0
    if y1_padding < 0:
        y1_padding = 0
    if x1_padding + w1_padding > real_w:
        w1_padding = real_w - x1_padding
    if y1_padding + h1_padding > real_h:
        h1_padding = real_h - y1_padding

    return x1_padding, y1_padding, w1_padding, h1_padding


def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:

        return None

    x1, y1, w, h = faces[0]

    # Get image dimensions
    real_w = img.shape[1]
    real_h = img.shape[0]


    area_image = real_h * real_w
    area_bbox = w * h
    ratio_bbox_and_image = area_bbox / area_image

    # Get padding indices for the bounding box
    x1_padding, y1_padding, w1_padding, h1_padding = get_padding_bbox_indices(
        x1, y1, w, h, real_w, real_h, ratio_bbox_and_image
    )

    # Crop the image around the face with padding
    padding_img = img[y1_padding:y1 + h1_padding, x1_padding:x1 + w1_padding]

    # Resize the image to the target dimensions
    resized_padding_img = cv2.resize(padding_img, dim, interpolation=cv2.INTER_AREA)

    # Normalize the pixel values to [0, 1]
    arr = np.asarray(resized_padding_img) / 255.0

    return arr


def load_and_preprocess_images(dataset_path):
    image_data = []
    labels = []

    for folder in ['ClientRaw', 'ImposterRaw']:
        folder_path = os.path.join(dataset_path, folder)
        label = 0 if folder == 'ClientRaw' else 1  # 0 for real faces, 1 for fake faces

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    img_path = os.path.join(subfolder_path, filename)

                    if img_path.endswith('.jpg'):
                        img = cv2.imread(img_path)

                        if img is not None:
                            preprocessed_img = preprocess(img)

                            if preprocessed_img is not None:
                                image_data.append(preprocessed_img)
                                labels.append(label)

    image_data = np.array(image_data)
    labels = np.array(labels)

    return image_data, labels


# Example usage
dataset_path = 'raw'
images, labels = load_and_preprocess_images(dataset_path)
import tensorflow as tf
import cv2

# Load the model from an HDF5 file
model = tf.keras.models.load_model('my_model.h5')
print(f"Processed {len(images)} images.")
loss, accuracy = model.evaluate(images, labels)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")