import cv2
from skimage.metrics import structural_similarity as ssim

def ssim_compare(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score

# Load images
identical_images = [cv2.imread(f'identical/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(identical_images)):
    for j in range(i+1, len(identical_images)):
        score = ssim_compare(identical_images[i], identical_images[j])
        if score == 1.0:
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical (SSIM: {score})")

# Load images
diff_images = [cv2.imread(f'different/image{i}.png') for i in range(10)]

# Compare images
for i in range(len(diff_images)):
    for j in range(i+1, len(diff_images)):
        score = ssim_compare(diff_images[i], diff_images[j])
        if score == 1.0:
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical (SSIM: {score})")