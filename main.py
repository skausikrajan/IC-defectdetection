import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files
import os

# Function to preprocess the images: resize, grayscale, and normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    # Resize image to standard size
    image = cv2.resize(image, (300, 300))
    # Normalize image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (e.g., (300, 300, 1))
    return image

# Function to load and preprocess datasets (good and faulty IC images)
def load_dataset(image_paths):
    images = []
    labels = []
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)
        labels.append(0 if "good" in path else 1)  # Label: 0 for Good, 1 for Faulty
    return np.array(images), np.array(labels)

# Build CNN model for image classification
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (Good or Faulty)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to upload and train the model
def upload_and_train():
    # Upload Good IC images (n images)
    print("Please upload the Good IC images (you can upload multiple images):")
    uploaded_good_ic = files.upload()
    good_ic_paths = list(uploaded_good_ic.keys())

    # Upload Faulty IC images (n images)
    print("Please upload the Faulty IC images (you can upload multiple images):")
    uploaded_faulty_ic = files.upload()
    faulty_ic_paths = list(uploaded_faulty_ic.keys())

    # Combine all image paths
    all_image_paths = good_ic_paths + faulty_ic_paths

    # Load dataset and preprocess
    images, labels = load_dataset(all_image_paths)
    images = images / 255.0  # Normalize images to [0, 1]

    # Split the dataset into train and test sets
    train_images = images[:int(0.8 * len(images))]
    train_labels = labels[:int(0.8 * len(labels))]
    test_images = images[int(0.8 * len(images)):]
    test_labels = labels[int(0.8 * len(labels)):]

    # Build and train the CNN model
    model = build_cnn_model()
    model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))  # Increased epochs

    # Save the trained model
    model.save('ic_detection_model.h5')

    print("Model training completed!")

# Function to predict if the input IC image is good or faulty
def predict(input_image_path):
    # Load and preprocess the input image
    input_image = preprocess_image(input_image_path)
    input_image = input_image / 255.0  # Normalize image to [0, 1]

    # Load the trained model
    model = tf.keras.models.load_model('ic_detection_model.h5')

    # Predict the input image
    prediction = model.predict(np.expand_dims(input_image, axis=0))
    print("Prediction score:", prediction[0])

    if prediction[0] > 0.5:
        print("The input image is a Faulty IC - Fault detected!")
    else:
        print("The input image is a Good IC - No faults detected!")

    # Display the input image
    print("Input Image:")
    cv2_imshow(input_image[0])

# Function to upload input image and detect
def upload_and_detect():
    # Upload the input IC image to be tested
    print("Please upload the Input IC image:")
    uploaded_input_ic = files.upload()
    if len(uploaded_input_ic) != 1:
        print("Error: Please upload exactly one Input IC image.")
        return
    input_ic_path = list(uploaded_input_ic.keys())[0]

    # Perform IC detection
    predict(input_ic_path)

# Run the upload, train, and detect process
# First, train the model
upload_and_train()

# After training, test the model
upload_and_detect()
