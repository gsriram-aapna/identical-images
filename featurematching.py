import cv2

def orb_compare(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)

# Load images
identical_images = [cv2.imread(f'identical/image{i}.png', 0) for i in range(10)]

# Compare images
for i in range(len(identical_images)):
    for j in range(i+1, len(identical_images)):
        matches = orb_compare(identical_images[i], identical_images[j])
        if matches > 100:  # Threshold for considering images as identical
            print(f"Image {i} and Image {j} are identical (Matches: {matches})")
        else:
            print(f"Image {i} and Image {j} are not identical (Matches: {matches})")
            
# Load images
diff_images = [cv2.imread(f'different/image{i}.png', 0) for i in range(10)]

# Compare images
for i in range(len(diff_images)):
    for j in range(i+1, len(diff_images)):
        matches = orb_compare(diff_images[i], diff_images[j])
        if matches > 100:  # Threshold for considering images as identical
            print(f"Image {i} and Image {j} are identical (Matches: {matches})")
        else:
            print(f"Image {i} and Image {j} are not identical (Matches: {matches})")
