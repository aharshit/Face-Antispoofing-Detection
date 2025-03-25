import numpy as np
import tensorflow as tf
import cv2

# Load the model from an HDF5 file
model = tf.keras.models.load_model('my_model.h5')
dim = (224, 224)


def get_padding_bbox_indices(x1, y1, w1, h1, real_w, real_h, ratio_bbox_and_image):
    x1_padding = x1 - int((w1) * (1 + ratio_bbox_and_image))
    y1_padding = y1 - int((h1) * (1 + ratio_bbox_and_image))
    w1_padding = w1 + int((w1) * (1 + ratio_bbox_and_image))
    h1_padding = h1 + int((h1) * (1 + ratio_bbox_and_image))
    if x1_padding < 0:
        x1_padding = 0
    if y1_padding < 0:
        y1_padding = 0
    if x1_padding + w1_padding > real_w:
        w1_padding = real_w - x1_padding
    if y1_padding + h1_padding > real_h:
        h1_padding = real_h - y1_padding
    return x1_padding, y1_padding, w1_padding, h1_padding


def preprocess(img, x1, y1, w, h):
    real_w = img.shape[1]
    real_h = img.shape[0]
    area_image = real_h * real_w
    area_bbox = w * h
    ratio_bbox_and_image = area_bbox / area_image
    x1_padding, y1_padding, w1_padding, h1_padding = get_padding_bbox_indices(
        x1, y1, w, h, real_w, real_h, ratio_bbox_and_image
    )
    padding_img = img[y1_padding:y1 + h1_padding, x1_padding:x1 + w1_padding]
    resized_padding_img = cv2.resize(padding_img, dim, interpolation=cv2.INTER_AREA)
    arr = np.asarray(resized_padding_img) / 255.0  # Normalize to [0, 1]
    return arr


# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera")
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            try:
                temp = preprocess(frame, x, y, w, h)
                img = np.expand_dims(temp, axis=0)  # Add batch dimension
                predicted_probabilities = model.predict(img)[0][0]  # Get scalar value

                if predicted_probabilities < 0.5:
                    color = (0, 0, 255)
                    ans = f"Spoof {100-(predicted_probabilities * 100):.2f}%"
                else:
                    color = (0, 255, 0)
                    ans = f"Live {predicted_probabilities * 100:.2f}%"

                thickness = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, ans, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Error processing face: {e}")

        # Display the frame with bounding boxes
        cv2.imshow('Face Detection', frame)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
predict_image()