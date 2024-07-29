import cv2
import numpy as np
from PIL import Image

def ncc_compare(img1, img2):
    img1 = np.array(img1.convert('L'))  # Convert image to grayscale and to numpy array
    img2 = np.array(img2.convert('L'))  # Convert image to grayscale and to numpy array
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        return False
    
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val == 1.0

# Load images
identical_images = [Image.open(f'identical/image{i}.png') for i in range(10)]

# Compare images using NCC
for i in range(len(identical_images)):
    for j in range(i+1, len(identical_images)):
        if ncc_compare(identical_images[i], identical_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")

# Load images
diff_images = [Image.open(f'different/image{i}.png') for i in range(10)]

# Compare images using NCC
for i in range(len(diff_images)):
    for j in range(i+1, len(diff_images)):
        if ncc_compare(diff_images[i], diff_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")

