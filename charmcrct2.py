import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os
import matplotlib.pyplot as plt
import pytesseract

# 🔹 Set path to Tesseract OCR (Update this based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 Define dataset paths
TRAIN_GOOD_IC_PATH = "C:\\Upd Dataset\\Good ic 2"
TRAIN_SCRATCH_IC_PATH = "C:\\Upd Dataset\\Scratch 2"
TRAIN_BROKEN_IC_PATH = "C:\\Upd Dataset\\Broken 2"

VAL_GOOD_IC_PATH = "C:\\Upd Dataset\\good ic 3"
VAL_SCRATCH_IC_PATH = "C:\\Upd Dataset\\scratch 3"
VAL_BROKEN_IC_PATH = "C:\\Upd Dataset\\broken 3"

CLASS_LABELS = {"good": 0, "scratch": 1, "broken": 2}

# 🔹 Preprocess Images: Resize, Normalize, Convert to RGB
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (224, 224))  # Resize
    image = np.expand_dims(image, axis=-1)  # Expand dimensions (H, W, 1)
    image = np.repeat(image, 3, axis=-1)  # Convert to 3 channels
    return image / 255.0  # Normalize

# 🔹 Load Dataset
def load_dataset(good_ic_path, scratch_ic_path, broken_ic_path):
    images, labels = [], []

    for file in os.listdir(good_ic_path):
        image_path = os.path.join(good_ic_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        images.append(preprocess_image(image))
        labels.append(0)  # Good IC

    for file in os.listdir(scratch_ic_path):
        image_path = os.path.join(scratch_ic_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        images.append(preprocess_image(image))
        labels.append(1)  # Scratched IC

    for file in os.listdir(broken_ic_path):
        image_path = os.path.join(broken_ic_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        images.append(preprocess_image(image))
        labels.append(2)  # Broken IC

    return np.array(images), np.array(labels)

# 🔹 Data Augmentation
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True, vertical_flip=True
    )
    return datagen.flow(train_images, train_labels, batch_size=16)

# 🔹 Build Optimized MobileNetV2 Model
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')  # 3 output classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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

# 🔹 Train the Model
def train_model():
    print("📂 Loading dataset...")

    train_images, train_labels = load_dataset(TRAIN_GOOD_IC_PATH, TRAIN_SCRATCH_IC_PATH, TRAIN_BROKEN_IC_PATH)
    val_images, val_labels = load_dataset(VAL_GOOD_IC_PATH, VAL_SCRATCH_IC_PATH, VAL_BROKEN_IC_PATH)

    if len(train_images) == 0 or len(val_images) == 0:
        print("❌ Error: Training or validation dataset is empty.")
        return

    print(f"✅ Loaded {len(train_images)} training images and {len(val_images)} validation images.")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_cnn_model()
    augmented_data = augment_data(train_images, train_labels)

    checkpoint = ModelCheckpoint('ic_fault_detector_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(
        augmented_data,
        validation_data=(val_images, val_labels),
        epochs=50,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )

    plot_training_history(history)
    model.save('ic_fault_detector_model.keras')
    print("✅ Model training complete.")

# 🔹 Function to Detect Text in an IC
def contains_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')  # Extract text
    return bool(text.strip())  # Returns True if text is found

# 🔹 Real-time IC Defect Detection using Webcam
def real_time_detection():
    print("\n🎥 Initializing Webcam for Real-Time Detection...")
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')
    fault_types = ['Good IC', 'Scratched IC', 'Broken Legs']

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to read frame.")
            break

        # Check if detected "scratch" is actually text
        if contains_text(frame):
            label = "IC Status: Good IC (Text Detected)"
        else:
            processed_frame = preprocess_image(frame)
            prediction = model.predict(np.expand_dims(processed_frame, axis=0))
            predicted_class = np.argmax(prediction, axis=1)[0]
            fault_type = fault_types[predicted_class]
            label = f"IC Status: {fault_type}"

        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("IC Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔹 Main Execution
if __name__ == "__main__":
    train_model()
    real_time_detection()
