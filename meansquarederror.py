import numpy as np
from PIL import Image

# Function to calculate Mean Squared Error (MSE) between two images
def mse_compare(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    if img1.shape != img2.shape:
        return False
    err = np.sum((img1 - img2) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err == 0

# Load images
identical_images = [Image.open(f'identical/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(identical_images)):
    for j in range(i+1, len(identical_images)):
        if mse_compare(identical_images[i], identical_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")

# Load images
diff_images = [Image.open(f'different/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(diff_images)):
    for j in range(i+1, len(diff_images)):
        if mse_compare(diff_images[i], diff_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")
