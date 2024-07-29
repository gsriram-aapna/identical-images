from PIL import Image
import numpy as np

def pixel_compare(img1, img2):
    return np.array_equal(img1, img2)

# Load images
identical_images = [Image.open(f'identical/image{i}.png') for i in range(10)]
identical_image_arrays = [np.array(img) for img in identical_images]

# Compare images
for i in range(len(identical_image_arrays)):
    for j in range(i+1, len(identical_image_arrays)):
        if pixel_compare(identical_image_arrays[i], identical_image_arrays[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")
            
# Load images
diff_images = [Image.open(f'different/image{i}.png') for i in range(10)]
diff_image_arrays = [np.array(img) for img in diff_images]

# Compare images
for i in range(len(diff_image_arrays)):
    for j in range(i+1, len(diff_image_arrays)):
        if pixel_compare(diff_image_arrays[i], diff_image_arrays[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")

