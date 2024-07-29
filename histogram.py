import cv2

def histogram_compare(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# Load image
identical_image = [cv2.imread(f'identical/image{i}.png', 0) for i in range(10)]

# Compare image
for i in range(len(identical_image)):
    for j in range(i+1, len(identical_image)):
        score = histogram_compare(identical_image[i], identical_image[j])
        if score == 1.0:
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical (Histogram Correlation: {score})")

# Load image
diff_image = [cv2.imread(f'different/image{i}.png', 0) for i in range(10)]

# Compare image
for i in range(len(diff_image)):
    for j in range(i+1, len(diff_image)):
        score = histogram_compare(diff_image[i], diff_image[j])
        if score == 1.0:
            print(f"Image {i} and Image {j} are identical")
        else:
            print(f"Image {i} and Image {j} are not identical (Histogram Correlation: {score})")
