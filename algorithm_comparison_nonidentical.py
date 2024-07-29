import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import imagehash
from sklearn.metrics import precision_score, recall_score, f1_score

# Load images (assuming they are named image0.png, image1.png, etc.)
images = [Image.open(f'different/image{i}.png') for i in range(10)]

# Ground truth list
ground_truth = [
    # Identical pairs
    (0, 1, True),
    (2, 3, True),
    # Non-identical pairs
    (0, 2, False), (0, 3, False), (0, 4, False), (0, 5, False), (0, 6, False), (0, 7, False), (0, 8, False), (0, 9, False),
    (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, False), (1, 6, False), (1, 7, False), (1, 8, False), (1, 9, False),
    (2, 4, False), (2, 5, False), (2, 6, False), (2, 7, False), (2, 8, False), (2, 9, False),
    (3, 4, False), (3, 5, False), (3, 6, False), (3, 7, False), (3, 8, False), (3, 9, False),
    (4, 5, False), (4, 6, False), (4, 7, False), (4, 8, False), (4, 9, False),
    (5, 6, False), (5, 7, False), (5, 8, False), (5, 9, False),
    (6, 7, False), (6, 8, False), (6, 9, False),
    (7, 8, False), (7, 9, False),
    (8, 9, False)
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