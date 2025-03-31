import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os
import matplotlib.pyplot as plt

# Dataset paths
TRAIN_PATHS = {
    "good": "C:\\Upd Dataset\\Good ic 2",
    "scratch": "C:\\Upd Dataset\\Scratch 2",
    "broken": "C:\\Upd Dataset\\Broken 2"
}

VAL_PATHS = {
    "good": "C:\\Upd Dataset\\good ic 3",
    "scratch": "C:\\Upd Dataset\\scratch 3",
    "broken": "C:\\Upd Dataset\\broken 3"
}

CLASS_LABELS = {"good": 0, "scratch": 1, "broken": 2}


# Preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    return image / 255.0


# Load dataset
def load_dataset(paths):
    images, labels = [], []
    for label, path in paths.items():
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            images.append(preprocess_image(image))
            labels.append(CLASS_LABELS[label])
    return np.array(images), np.array(labels)


# Data augmentation
def augment_data(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True, vertical_flip=True
    )
    return datagen.flow(images, labels, batch_size=16)


# Build MobileNetV2 model
def build_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


# Train model
def train_model():
    print("📂 Loading dataset...")
    train_images, train_labels = load_dataset(TRAIN_PATHS)
    val_images, val_labels = load_dataset(VAL_PATHS)

    if len(train_images) == 0 or len(val_images) == 0:
        print("❌ Error: Dataset is empty. Check paths.")
        return

    print(f"✅ Loaded {len(train_images)} training and {len(val_images)} validation images.")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_model()
    augmented_data = augment_data(train_images, train_labels)

    callbacks = [
        ModelCheckpoint('ic_fault_detector_model.keras', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=7),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        augmented_data,
        validation_data=(val_images, val_labels),
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    plot_history(history)
    model.save('ic_fault_detector_model.keras')
    print("✅ Model training complete.")


# Real-time IC detection
def real_time_detection():
    print("🎥 Starting Real-Time Detection...")
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')
    fault_types = ['Good IC', 'Scratched IC', 'Broken Legs']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam error.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame error.")
            break

        processed_frame = preprocess_image(frame)
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100

        label = f"IC Status: {fault_types[predicted_class]} ({confidence:.2f}%)"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("IC Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔄 Detection stopped.")


if __name__ == "__main__":
    print("🚀 Training Model...")
    train_model()
    print("🎥 Starting Real-Time Detection...")
    real_time_detection()
