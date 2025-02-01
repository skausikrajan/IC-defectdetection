import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow
from google.colab import files
import matplotlib.pyplot as plt

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
    bboxes = []  # Bounding boxes for defects
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)
        if "good" in path.lower():
            labels.append(0)  # Good
            bboxes.append([0, 0, 0, 0])  # No bounding box for good images
        elif "scratch" in path.lower():
            labels.append(1)  # Scratch
            bboxes.append([0.2, 0.2, 0.8, 0.8])  # Example bounding box for scratch
        elif "broken" in path.lower():
            labels.append(2)  # Broken Legs
            bboxes.append([0.1, 0.1, 0.9, 0.9])  # Example bounding box for broken legs
    return np.array(images), np.array(labels), np.array(bboxes)

# IoU loss function for bounding box regression
def iou_loss(y_true, y_pred):
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    x1_int = tf.maximum(x1_true, x1_pred)
    y1_int = tf.maximum(y1_true, y1_pred)
    x2_int = tf.minimum(x2_true, x2_pred)
    y2_int = tf.minimum(y2_true, y2_pred)

    intersection = tf.maximum(0.0, x2_int - x1_int) * tf.maximum(0.0, y2_int - y1_int)
    
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union = true_area + pred_area - intersection

    iou = intersection / (union + 1e-6)
    return 1 - iou

# Build a CNN model with MobileNetV2 base and additional layers
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model layers

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    class_output = layers.Dense(3, activation='softmax', name="class_output")(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name="bbox_output")(x)  # Ensure bbox is within 0-1 range

    model = models.Model(inputs, outputs=[class_output, bbox_output])
    
    model.compile(optimizer='adam', 
                  loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': iou_loss},
                  metrics={'class_output': 'accuracy', 'bbox_output': 'mse'})
    return model

# Function to plot training accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['class_output_accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_class_output_accuracy'], label='Validation Accuracy', marker='o')
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
    images, labels, bboxes = load_dataset(all_image_paths)
    
    train_images, val_images, train_labels, val_labels, train_bboxes, val_bboxes = train_test_split(
        images, labels, bboxes, test_size=0.2, random_state=42
    )
    
    model = build_cnn_model()
    
    # Callbacks for early stopping and saving the best model
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(
        train_images, {'class_output': train_labels, 'bbox_output': train_bboxes},
        validation_data=(val_images, {'class_output': val_labels, 'bbox_output': val_bboxes}),
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )
    
    plot_training_history(history)
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")

# Function to predict faults in new images
def predict_fault():
    print("Please upload the images for fault detection:")
    uploaded_files = files.upload()
    image_paths = list(uploaded_files.keys())
    
    model = tf.keras.models.load_model('ic_fault_detector_model.keras', custom_objects={'iou_loss': iou_loss})
    fault_types = ['Good', 'Scratch', 'Broken Legs']
    
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        class_pred, bbox_pred = model.predict(image)
        predicted_class = np.argmax(class_pred, axis=1)[0]
        fault_type = fault_types[predicted_class]
        bbox = bbox_pred[0] * 224  # Convert normalized coordinates to pixel values

        # Convert bounding box to pixel coordinates
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, fault_type, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2_imshow(original_image)
        print(f"Fault Detected: {fault_type}")
        print("Confidence Scores:")
        for i, fault in enumerate(fault_types):
            print(f"{fault}: {class_pred[0][i] * 100:.2f}%")

# Main Workflow
upload_and_train()
predict_fault()
