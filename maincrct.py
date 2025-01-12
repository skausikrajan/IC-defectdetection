import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files

# Function to preprocess the images: resize, grayscale, and normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    # Resize image to standard size
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image / 255.0  # Normalize image to [0, 1]

# Function to load and preprocess datasets
def load_dataset(image_paths):
    images = []
    labels = []  # 0: Good, 1: Scratch, 2: Broken Legs
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)

        # Assign labels based on file naming conventions
        if "good" in path.lower():
            labels.append(0)  # Good
        elif "scratch" in path.lower():
            labels.append(1)  # Scratch
        elif "broken" in path.lower():
            labels.append(2)  # Broken Legs

    return np.array(images), np.array(labels)

# Build CNN model for multi-class classification
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: Good, Scratch, Broken
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to upload and train the model
def upload_and_train():
    print("Please upload the Good IC images:")
    uploaded_good_ic = files.upload()
    good_ic_paths = list(uploaded_good_ic.keys())

    print("Please upload the Scratch Fault IC images:")
    uploaded_scratch_ic = files.upload()
    scratch_ic_paths = list(uploaded_scratch_ic.keys())

    print("Please upload the Broken Legs IC images:")
    uploaded_broken_ic = files.upload()
    broken_ic_paths = list(uploaded_broken_ic.keys())

    # Combine and preprocess datasets
    all_image_paths = good_ic_paths + scratch_ic_paths + broken_ic_paths
    images, labels = load_dataset(all_image_paths)

    # Split dataset into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Build and train the CNN model
    model = build_cnn_model()
    model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=30, batch_size=32
    )

    # Save the trained model
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")

# Function to predict faults in new images
def predict_fault():
    print("Please upload the images for fault detection:")
    uploaded_files = files.upload()
    image_paths = list(uploaded_files.keys())

    # Load the trained model
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')

    # Define fault types
    fault_types = ['Good', 'Scratch', 'Broken Legs']

    # Predict each uploaded image
    for image_path in image_paths:
        # Display the original image
        original_image = cv2.imread(image_path)
        cv2_imshow(original_image)

        # Preprocess the image
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the CNN model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]

        # Display the result
        print(f"Fault Detected: {fault_type}")

# Main Workflow
# 1. Train the model
upload_and_train()

# 2. Predict faults in new images
predict_fault()
