import numpy as np

def extract_patches_with_labels(X, y):

    num_images = X.shape[0]
    h, w, c = X.shape[1:]
    h_step, w_step = h // 3, w // 3

    patches = np.zeros((num_images, 9, h_step, w_step, c), dtype=X.dtype)

    # Extract patches
    for i in range(num_images):
        img = X[i]

        patch_idx = 0
        # Divide the image into 3x3 grid of patches
        for row in range(3):
            for col in range(3):
                # Calculate patch coordinates
                h_start = row * h_step
                h_end = (row + 1) * h_step
                w_start = col * w_step
                w_end = (col + 1) * w_step

                # Extract patch and store it
                patches[i, patch_idx] = img[h_start:h_end, w_start:w_end, :]
                patch_idx += 1

    # Labels remain unchanged as they correspond to original images
    return patches, y


# Load the data from the .npz file
data = np.load('anti_spoofing_data.npz')
X = data['arr_0']  # Images
y = data['arr_1']  # Labels

# Extract patches and labels
X_patches, y_labels = extract_patches_with_labels(X, y)

# Print shapes of patches and labels
print("Patches shape:", X_patches.shape)
print("Labels shape:", y_labels.shape)

# Save the patches and labels to a new .npz file
np.savez('patches_with_labels.npz', X=X_patches, y=y_labels)
