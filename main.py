import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files

def detect_ic(input_image_path, good_ic_path, faulty_ic_path):
    # Load the input, good IC, and faulty IC images
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    good_ic_image = cv2.imread(good_ic_path, cv2.IMREAD_GRAYSCALE)
    faulty_ic_image = cv2.imread(faulty_ic_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or good_ic_image is None or faulty_ic_image is None:
        print("Error: Could not load one or more images. Please check the file paths.")
        return

    # Resize the images to the same size for comparison
    input_image = cv2.resize(input_image, (300, 300))
    good_ic_image = cv2.resize(good_ic_image, (300, 300))
    faulty_ic_image = cv2.resize(faulty_ic_image, (300, 300))

    # Compute the absolute differences
    diff_with_good = cv2.absdiff(input_image, good_ic_image)
    diff_with_faulty = cv2.absdiff(input_image, faulty_ic_image)

    # Threshold the differences to highlight significant changes
    _, thresh_good = cv2.threshold(diff_with_good, 50, 255, cv2.THRESH_BINARY)
    _, thresh_faulty = cv2.threshold(diff_with_faulty, 50, 255, cv2.THRESH_BINARY)

    # Count the non-zero pixels in the thresholded differences
    non_zero_good = cv2.countNonZero(thresh_good)
    non_zero_faulty = cv2.countNonZero(thresh_faulty)

    # Determine whether the input image matches the Good IC or Faulty IC more closely
    if non_zero_good < non_zero_faulty:
        print("The input image is a Good IC - No faults detected!")
    else:
        print("The input image is a Faulty IC - Fault detected!")

    # Display images for debugging (optional)
    print("Input Image:")
    cv2_imshow(input_image)
    print("Difference with Good IC:")
    cv2_imshow(diff_with_good)
    print("Difference with Faulty IC:")
    cv2_imshow(diff_with_faulty)

# Step 1: Upload the reference images (Good IC and Faulty IC)
print("Please upload the Good IC and Faulty IC reference images:")
uploaded = files.upload()
if len(uploaded) != 3:
    print("Error: Please upload exactly three images (Good IC, Faulty IC, and Input IC).")
else:
    # Get the file paths for Good IC, Faulty IC, and Input IC images
    file_paths = list(uploaded.keys())
    good_ic_path = file_paths[0]  # First uploaded image is Good IC
    faulty_ic_path = file_paths[1]  # Second uploaded image is Faulty IC
    input_ic_path = file_paths[2]  # Third uploaded image is Input IC

    # Step 2: Perform IC detection
    detect_ic(input_ic_path, good_ic_path, faulty_ic_path)
