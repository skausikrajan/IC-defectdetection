import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files
import matplotlib.pyplot as plt  # For accuracy and loss visualization

# Function to preprocess the images: resize, grayscale, and normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (224, 224))  # Resize to 224x224 for MobileNetV2
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale (1 channel)
    image = np.repeat(image, 3, axis=-1)  # Convert grayscale (1 channel) to RGB (3 channels)
    return image / 255.0  # Normalize to [0, 1]

# Function to load and preprocess datasets
def load_dataset(image_paths):
    images = []
    labels = []  # 0: Good, 1: Scratch, 2: Broken Legs
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)
        if "good" in path.lower():
            labels.append(0)  # Good
        elif "scratch" in path.lower():
            labels.append(1)  # Scratch
        elif "broken" in path.lower():
            labels.append(2)  # Broken Legs
    return np.array(images), np.array(labels)

# Data augmentation generator with all types of flips
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True  # Horizontal and vertical flips
    )

    augmented_images = []
    augmented_labels = []

    # Apply additional diagonal flips manually
    for image, label in zip(train_images, train_labels):
        augmented_images.append(image)
        augmented_labels.append(label)

        # Horizontal flip
        horizontal_flip = cv2.flip(image, 1)
        augmented_images.append(horizontal_flip)
        augmented_labels.append(label)

        # Vertical flip
        vertical_flip = cv2.flip(image, 0)
        augmented_images.append(vertical_flip)
        augmented_labels.append(label)

        # Diagonal flip (horizontal + vertical)
        diagonal_flip = cv2.flip(image, -1)
        augmented_images.append(diagonal_flip)
        augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    return datagen.flow(augmented_images, augmented_labels, batch_size=8)

# Build a CNN model with MobileNetV2 base and additional layers
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: Good, Scratch, Broken
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot training accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to upload, preprocess, and train the model
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
    
    all_image_paths = good_ic_paths + scratch_ic_paths + broken_ic_paths
    images, labels = load_dataset(all_image_paths)
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    model = build_cnn_model()
    
    augmented_data = augment_data(train_images, train_labels)

    # Callbacks for early stopping and saving the best model
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(
        augmented_data,
        validation_data=(val_images, val_labels),
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )
    
    plot_training_history(history)
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")

# Function to predict faults and display bounding boxes
def predict_fault():
    print("Please upload the images for fault detection:")
    uploaded_files = files.upload()
    image_paths = list(uploaded_files.keys())
    
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')
    fault_types = ['Good', 'Scratch', 'Broken Legs']
    
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        cv2_imshow(original_image)
        image = preprocess_image(image_path)
        image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image_expanded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]
        print(f"Fault Detected: {fault_type}")
        print("Confidence Scores:")
        for i, fault in enumerate(fault_types):
            print(f"{fault}: {prediction[0][i] * 100:.2f}%")
        
        # Visualizing bounding boxes
        if fault_type != "Good":  # If defect is detected, draw bounding box
            # Simulating bounding box around the detected defect area (using random values here for demonstration)
            height, width, _ = original_image.shape
            x1, y1 = int(width * 0.1), int(height * 0.1)  # Top left corner
            x2, y2 = int(width * 0.5), int(height * 0.5)  # Bottom right corner
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for defect
            cv2.putText(original_image, f"Defect: {fault_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the result with bounding box
        cv2_imshow(original_image)

# Main Workflow
upload_and_train()
predict_fault()
