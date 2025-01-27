import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from google.colab import files  # For uploading files
import matplotlib.pyplot as plt  # For accuracy and loss visualization

# Function to preprocess the images: resize, grayscale, and normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
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

# Data augmentation generator
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    return datagen.flow(train_images, train_labels, batch_size=8)

# Build an enhanced CNN model
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False  # Freeze the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: Good, Scratch, Broken
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM function to visualize the saliency map
def grad_cam(image, model, class_idx):
    # Use the last convolutional layer for Grad-CAM
    target_layer = model.get_layer('mobilenetv2_1.00_224').output  # Correct layer name
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[target_layer, model.output])
    
    # Record gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)

    # Compute the weights for Grad-CAM
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.dot(conv_outputs[0], weights)  # Apply the weights to the convolutional layer outputs
    cam = np.maximum(cam, 0)  # ReLU activation
    heatmap = cam / cam.max()  # Normalize

    # Resize to the original image size
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap

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
    train_images = np.repeat(train_images, 3, axis=-1)  # Convert grayscale to RGB
    val_images = np.repeat(val_images, 3, axis=-1)  # Convert grayscale to RGB
    train_images = np.array([cv2.resize(img, (224, 224)) for img in train_images])  # Resize to 224x224
    val_images = np.array([cv2.resize(img, (224, 224)) for img in val_images])  # Resize to 224x224

    model = build_cnn_model()
    augmented_data = augment_data(train_images, train_labels)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        augmented_data,
        validation_data=(val_images, val_labels),
        epochs=10,  # Reduced epochs due to small dataset
        callbacks=[early_stopping]
    )
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")
    plot_training_history(history)

# Function to predict faults in new images
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
        image = np.repeat(image, 3, axis=-1)  # Convert grayscale to RGB
        image = cv2.resize(image, (224, 224))  # Resize to 224x224
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]
        print(f"Fault Detected: {fault_type}")
        print("Confidence Scores:")
        for i, fault in enumerate(fault_types):
            print(f"{fault}: {prediction[0][i] * 100:.2f}%")

# Main Workflow
upload_and_train()
predict_fault()
