import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files

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
    labels = []  # 0 for Good, 1 for Faulty
    fault_types = []  # Additional labels for fault types
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)
        
        # Assigning labels based on image naming convention
        if "good" in path:
            labels.append(0)
            fault_types.append("Good")
        else:
            labels.append(1)
            # Assign fault type based on file name
            if "surface" in path:
                fault_types.append("Surface Scratches")
            elif "pin" in path:
                fault_types.append("Pin Defect")
            else:
                fault_types.append("Unknown")
    
    return np.array(images), np.array(labels), fault_types

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
        layers.Dense(2, activation='softmax')  # Output two classes: Good or Faulty
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    images, labels, fault_types = load_dataset(all_image_paths)
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
    input_image_display = cv2.imread(input_image_path)  # Load the original image for display
    input_image = input_image / 255.0  # Normalize image to [0, 1]

    # Load the trained model
    model = tf.keras.models.load_model('ic_detection_model.h5')

    # Predict the input image
    prediction = model.predict(np.expand_dims(input_image, axis=0))
    print("Prediction scores:", prediction[0])

    # Classify as Good or Faulty
    if np.argmax(prediction) == 0:
        print("The input image is a Good IC - No faults detected!")
    else:
        print("The input image is a Faulty IC - Fault detected!")
        
        # Identify the fault type based on file naming convention
        if "surface" in input_image_path:
            print("Fault Type: Surface Scratches")
        elif "pin" in input_image_path:
            print("Fault Type: Pin Defect")
        else:
            print("Fault Type: Unknown")

    # Display the input image
    print("Input Image:")
    cv2_imshow(input_image_display)

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
