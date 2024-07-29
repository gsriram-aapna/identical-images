import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import imagehash
from sklearn.metrics import precision_score, recall_score, f1_score

# Load images
images = [Image.open(f'identical/image{i}.png') for i in range(10)]

# Ground truth list
ground_truth = [
    (0, 1, True),
    (0, 2, True),
    (0, 3, True),
    (0, 4, True),
    (0, 5, True),
    (0, 6, True),
    (0, 7, True),
    (0, 8, True),
    (0, 9, True),
    (1, 2, True),
    (1, 3, True),
    (1, 4, True),
    (1, 5, True),
    (1, 6, True),
    (1, 7, True),
    (1, 8, True),
    (1, 9, True),
    (2, 3, True),
    (2, 4, True),
    (2, 5, True),
    (2, 6, True),
    (2, 7, True),
    (2, 8, True),
    (2, 9, True),
    (3, 4, True),
    (3, 5, True),
    (3, 6, True),
    (3, 7, True),
    (3, 8, True),
    (3, 9, True),
    (4, 5, True),
    (4, 6, True),
    (4, 7, True),
    (4, 8, True),
    (4, 9, True),
    (5, 6, True),
    (5, 7, True),
    (5, 8, True),
    (5, 9, True),
    (6, 7, True),
    (6, 8, True),
    (6, 9, True),
    (7, 8, True),
    (7, 9, True),
    (8, 9, True)
]


# Function to evaluate a comparison algorithm
def evaluate_algorithm(compare_function):
    y_true = []
    y_pred = []
    for img1_idx, img2_idx, are_identical in ground_truth:
        y_true.append(are_identical)
        result = compare_function(images[img1_idx], images[img2_idx])
        y_pred.append(result)
    
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=1)
    recall = recall_score(y_true, y_pred, pos_label=True, zero_division=1)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=1)
    return precision, recall, f1

# Comparison algorithms
def pixel_compare(img1, img2):
    return np.array_equal(np.array(img1), np.array(img2))

def hash_compare(img1, img2):
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return hash1 == hash2

def ssim_compare(img1, img2):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score == 1.0

def histogram_compare(img1, img2):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score == 1.0

def orb_compare(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(np.array(img1), None)
    kp2, des2 = orb.detectAndCompute(np.array(img2), None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches) > 100  # Arbitrary threshold for identical images

def mse_compare(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    err = np.sum((img1 - img2) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err == 0

def ncc_compare(img1, img2):
    img1 = np.array(img1.convert('L'))  # Convert image to grayscale and to numpy array
    img2 = np.array(img2.convert('L'))  # Convert image to grayscale and to numpy array
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        return False
    
    result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val == 1.0

# Evaluate algorithms
algorithms = [pixel_compare, hash_compare, ssim_compare, histogram_compare, orb_compare, mse_compare, ncc_compare]

for compare_func in algorithms:
    precision, recall, f1 = evaluate_algorithm(compare_func)
    print(f"{compare_func.__name__} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
