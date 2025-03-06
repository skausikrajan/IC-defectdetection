import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os
import matplotlib.pyplot as plt

# 🔹 Define dataset paths
TRAIN_GOOD_IC_PATH = "C:\\Upd Dataset\\Good ic 2"
TRAIN_SCRATCH_IC_PATH = "C:\\Upd Dataset\\Scratch 2"
TRAIN_BROKEN_IC_PATH = "C:\\Upd Dataset\\Broken 2"

VAL_GOOD_IC_PATH = "C:\\Upd Dataset\\good ic 3"
VAL_SCRATCH_IC_PATH = "C:\\Upd Dataset\\scratch 3"
VAL_BROKEN_IC_PATH = "C:\\Upd Dataset\\broken 3"

CLASS_LABELS = {"good": 0, "scratch": 1, "broken": 2}


# 🔹 Preprocess Images
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (224, 224))  # Resize
    image = np.expand_dims(image, axis=-1)  # Expand dimensions (H, W, 1)
    image = np.repeat(image, 3, axis=-1)  # Convert to 3 channels
    return image / 255.0  # Normalize


# 🔹 Load Dataset
def load_dataset(good_ic_path, scratch_ic_path, broken_ic_path):
    images, labels = [], []
    paths_labels = [(good_ic_path, 0), (scratch_ic_path, 1), (broken_ic_path, 2)]

    for path, label in paths_labels:
        for file in os.listdir(path):
            image_path = os.path.join(path, file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            images.append(preprocess_image(image))
            labels.append(label)

    return np.array(images), np.array(labels)


# 🔹 Data Augmentation
def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True
    )
    return datagen.flow(train_images, train_labels, batch_size=16)


# 🔹 Build Optimized Model
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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 🔹 Plot Training Graph
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
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

    # Compute Class Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_cnn_model()
    augmented_data = augment_data(train_images, train_labels)

    # Callbacks
    checkpoint = ModelCheckpoint('ic_fault_detector_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
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


# 🔹 Real-time IC Defect Detection using Webcam
def real_time_detection():
    print("\n🎥 Initializing Webcam for Real-Time Detection...")
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')
    fault_types = ['Good IC', 'Scratched IC', 'Broken Legs']

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Webcam not accessible.")
        return

    prev_label = None
    no_object_threshold = 30
    no_object_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            no_object_counter += 1
            if no_object_counter > no_object_threshold:
                prev_label = None
            cv2.imshow("IC Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        no_object_counter = 0

        processed_frame = preprocess_image(frame)
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]
        confidence = prediction[0][predicted_class] * 100

        if prev_label != fault_type:
            prev_label = fault_type

        label = f"IC Status: {fault_type} ({confidence:.2f}%)"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("IC Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔄 Real-time detection stopped.")


# 🔹 Main Execution
if __name__ == "__main__":
    print("\n🚀 Training Model...")
    train_model()

    print("\n🎥 Starting Real-Time Detection...")
    real_time_detection()
