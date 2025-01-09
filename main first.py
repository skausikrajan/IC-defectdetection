import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files
from skimage.metrics import structural_similarity as ssim  # For similarity measurement

def preprocess_image(image_path):
    """
    Load and preprocess the image: grayscale, resize, enhance, and normalize.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    # Resize to a standard size
    image = cv2.resize(image, (300, 300))
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    # Normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image

def compute_similarity(input_image, reference_image):
    """
    Compute similarity metrics (SSIM and pixel-wise absolute difference).
    """
    # SSIM score and difference image
    ssim_score, ssim_diff = ssim(input_image, reference_image, full=True)
    ssim_diff = (ssim_diff * 255).astype(np.uint8)

    # Pixel-wise absolute difference
    abs_diff = cv2.absdiff(input_image, reference_image)
    non_zero_diff = np.count_nonzero(abs_diff)

    return ssim_score, ssim_diff, non_zero_diff

def detect_ic(input_image_path, good_ic_path, faulty_ic_path):
    try:
        # Preprocess the images
        input_image = preprocess_image(input_image_path)
        good_ic_image = preprocess_image(good_ic_path)
        faulty_ic_image = preprocess_image(faulty_ic_path)
    except ValueError as e:
        print(e)
        return

    # Compute similarity with Good IC
    score_good, diff_good, non_zero_good = compute_similarity(input_image, good_ic_image)

    # Compute similarity with Faulty IC
    score_faulty, diff_faulty, non_zero_faulty = compute_similarity(input_image, faulty_ic_image)

    print(f"SSIM with Good IC: {score_good:.4f}, Non-zero Diff: {non_zero_good}")
    print(f"SSIM with Faulty IC: {score_faulty:.4f}, Non-zero Diff: {non_zero_faulty}")

    # Combine metrics for a weighted decision
    weight_ssim = 0.7  # Weight for SSIM
    weight_diff = 0.3  # Weight for pixel difference (lower is better)

    score_good_combined = weight_ssim * score_good - weight_diff * non_zero_good
    score_faulty_combined = weight_ssim * score_faulty - weight_diff * non_zero_faulty

    print(f"Combined Score (Good IC): {score_good_combined:.4f}")
    print(f"Combined Score (Faulty IC): {score_faulty_combined:.4f}")

    # Decision based on combined scores
    if score_good_combined > score_faulty_combined:
        print("The input image is a Good IC - No faults detected!")
    else:
        print("The input image is a Faulty IC - Fault detected!")

    # Debugging: Display the images
    print("Input Image:")
    cv2_imshow(input_image)
    print("Difference with Good IC:")
    cv2_imshow(diff_good)
    print("Difference with Faulty IC:")
    cv2_imshow(diff_faulty)

# Upload images one at a time
def upload_and_detect():
    print("Please upload the Good IC reference image:")
    uploaded_good_ic = files.upload()
    if len(uploaded_good_ic) != 1:
        print("Error: Please upload exactly one Good IC image.")
        return
    good_ic_path = list(uploaded_good_ic.keys())[0]

    print("Please upload the Faulty IC reference image:")
    uploaded_faulty_ic = files.upload()
    if len(uploaded_faulty_ic) != 1:
        print("Error: Please upload exactly one Faulty IC image.")
        return
    faulty_ic_path = list(uploaded_faulty_ic.keys())[0]

    print("Please upload the Input IC image:")
    uploaded_input_ic = files.upload()
    if len(uploaded_input_ic) != 1:
        print("Error: Please upload exactly one Input IC image.")
        return
    input_ic_path = list(uploaded_input_ic.keys())[0]

    # Perform IC detection
    detect_ic(input_ic_path, good_ic_path, faulty_ic_path)

# Run the upload and detection process
upload_and_detect()
