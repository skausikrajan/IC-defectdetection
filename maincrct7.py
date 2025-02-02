import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from google.colab.patches import cv2_imshow  # Display images in Colab
from google.colab import files  # Upload files
import matplotlib.pyplot as plt  # Visualization

# 🔹 Preprocess Images: Resize, Grayscale → RGB, Normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (224, 224))  # Resize for MobileNetV2
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.repeat(image, 3, axis=-1)  # Convert grayscale → RGB
    return image / 255.0  # Normalize

# 🔹 Load & Preprocess Dataset
def load_dataset(image_paths):
    images, labels = [], []
    for path in image_paths:
        full_path = os.path.join("/content", path)  # Ensure full path
        image = preprocess_image(full_path)
        images.append(image)

        if "good" in path.lower():
            labels.append(0)  # Good
        elif "scratch" in path.lower():
            labels.append(1)  # Scratch
        elif "broken" in path.lower():
            labels.append(2)  # Broken Legs

    return np.array(images), np.array(labels)

# 🔹 Data Augmentation for Training
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
        zoom_range=0.05, horizontal_flip=False, vertical_flip=False
    )
    return datagen.flow(train_images, train_labels, batch_size=8)

# 🔹 Build MobileNetV2 Model
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 🔹 Plot Training History
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

# 🔹 Upload Dataset & Train Model
def upload_and_train():
    print("Please upload Good IC images:")
    uploaded_good_ic = files.upload()
    good_ic_paths = list(uploaded_good_ic.keys())

    print("Please upload Scratch Fault IC images:")
    uploaded_scratch_ic = files.upload()
    scratch_ic_paths = list(uploaded_scratch_ic.keys())

    print("Please upload Broken Legs IC images:")
    uploaded_broken_ic = files.upload()
    broken_ic_paths = list(uploaded_broken_ic.keys())

    all_image_paths = good_ic_paths + scratch_ic_paths + broken_ic_paths
    images, labels = load_dataset(all_image_paths)

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = build_cnn_model()
    augmented_data = augment_data(train_images, train_labels)

    # Compute Class Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Callbacks for Model Checkpoint & Early Stopping
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(
        augmented_data,
        validation_data=(val_images, val_labels),
        epochs=50,
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weight_dict
    )

    plot_training_history(history)
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")

# 🔹 Predict Faults & Draw Bounding Boxes
def predict_fault():
    print("Please upload images for fault detection:")
    uploaded_files = files.upload()
    image_paths = list(uploaded_files.keys())

    model = tf.keras.models.load_model('ic_fault_detector_model.keras')

    # Make a dummy call to model to ensure it's initialized
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input)

    fault_types = ['Good', 'Scratch', 'Broken Legs']
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        image = preprocess_image(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        prediction = model.predict(image_expanded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]
        confidence = prediction[0][predicted_class] * 100

        if fault_type != "Good":
            height, width, _ = original_image.shape
            x1, y1 = int(width * 0.3), int(height * 0.3)
            x2, y2 = int(width * 0.7), int(height * 0.7)
            color = colors[predicted_class]

            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(original_image, f"{fault_type} ({confidence:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2_imshow(original_image)
        print(f"Fault Detected: {fault_type} with {confidence:.2f}% confidence")

# 🔹 Main Execution
upload_and_train()
predict_fault()
