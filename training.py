import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data in chunks
def load_data_efficiently(filename, chunk_size=1000):
    data = np.load(filename, mmap_mode='r')
    X, y = data['arr_0'], data['arr_1']
    return X, y


# Custom data generator
class MemoryEfficientDataGenerator:
    def __init__(self, X, y, batch_size=32, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(X)
        self.augment = augment
        if augment:
            self.datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )
        else:
            self.datagen = ImageDataGenerator()

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        while True:
            indices = np.random.permutation(self.n_samples)
            for i in range(0, self.n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = self.X[batch_indices].astype('float32') / 255.0  # Normalize here
                y_batch = self.y[batch_indices]

                if self.augment:
                    for X_aug, y_aug in self.datagen.flow(X_batch, y_batch, batch_size=len(X_batch), shuffle=False):
                        yield X_aug, y_aug
                        break
                else:
                    yield X_batch, y_batch


# Load and prepare data
print("Loading data...")
X, y = load_data_efficiently('anti_spoofing_data.npz')

# Print class distribution
check_live_label = np.sum(y == 1)
check_spoof_label = np.sum(y == 0)
print(f"There are 2 classes, number of live is {check_live_label} and number of spoof is {check_spoof_label}")

# Split indices instead of data
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, valid_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

print(f"Training data size: {len(train_idx)}")
print(f"Validation data size: {len(valid_idx)}")
print(f"Test data size: {len(test_idx)}")

# Create generators
batch_size = 32
train_generator = MemoryEfficientDataGenerator(
    X[train_idx], y[train_idx], batch_size=batch_size, augment=True)
valid_generator = MemoryEfficientDataGenerator(
    X[valid_idx], y[valid_idx], batch_size=batch_size, augment=False)
test_generator = MemoryEfficientDataGenerator(
    X[test_idx], y[test_idx], batch_size=batch_size, augment=False)

# Define the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
steps_per_epoch = len(train_idx) // batch_size
validation_steps = len(valid_idx) // batch_size

history = model.fit(
    iter(train_generator),
    validation_data=iter(valid_generator),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5
)

# Save the Model
model.save("./my_model.h5")

# Evaluate on Test Data
test_steps = len(test_idx) // batch_size
test_results = model.evaluate(iter(test_generator), steps=test_steps)
print(f"Test Accuracy: {test_results[1] * 100:.2f}%")

# Generate predictions in batches
y_pred = []
y_true = []
test_data = MemoryEfficientDataGenerator(X[test_idx], y[test_idx], batch_size=batch_size)
for i in range(len(test_idx) // batch_size):
    X_batch, y_batch = next(iter(test_data))
    pred_batch = (model.predict(X_batch) > 0.5).astype("int32").flatten()
    y_pred.extend(pred_batch)
    y_true.extend(y_batch)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Calculate Metrics
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
HTER = (FPR + FNR) / 2

print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"False Negative Rate (FNR): {FNR:.2f}")
print(f"Half Total Error Rate (HTER): {HTER:.2f}")
training_loss = history.history['loss']
accuracy = history.history['accuracy']
epochs = range(1, len(training_loss) + 1)

# Plot training and validation loss over epochs
plt.plot(epochs, training_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'r', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()