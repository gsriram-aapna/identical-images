import imagehash
from PIL import Image

def hash_compare(img1, img2):
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return hash1 == hash2

# Load images
identical_images = [Image.open(f'identical/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(identical_images)):
    for j in range(i+1, len(identical_images)):
        if hash_compare(identical_images[i], identical_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")

# Load images
diff_images = [Image.open(f'different/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(diff_images)):
    for j in range(i+1, len(diff_images)):
        if hash_compare(diff_images[i], diff_images[j]):
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical")