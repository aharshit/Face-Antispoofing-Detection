import numpy as np
import matplotlib.pyplot as plt

# Load the patches and labels from the 'patches_with_labels.npz' file
data = np.load('patches_with_labels.npz')
X_patches = data['X']  # Extracted patches (shape: (num_images * 9, patch_size[0], patch_size[1], channels))
y_patches = data['y']  # Corresponding labels (shape: (num_images * 9,))
print(y_patches.shape)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
# Loop through the axes and plot the images
for i, ax in enumerate(axes.flat):
    if i < 9:  # Ensure there are enough images
        ax.imshow(X_patches[9000][i])  # Display each image
        ax.axis('off')  # Hide axis labels
    else:
        ax.axis('off')  # Hide unused axes if less than 9 images

plt.tight_layout()
plt.show()