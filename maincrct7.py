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
from skimage.metrics import structural_similarity as ssim  

# Function to preprocess the images: resize, grayscale, and normalize
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (224, 224))  
    image = np.expand_dims(image, axis=-1)  
    image = np.repeat(image, 3, axis=-1)  
    return image / 255.0  

# Function to load and preprocess datasets
def load_dataset(image_paths):
    images = []
    labels = []  
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image)
        if "good" in path.lower():
            labels.append(0)  
        elif "scratch" in path.lower():
            labels.append(1)  
        elif "broken" in path.lower():
            labels.append(2)  
    return np.array(images), np.array(labels)

# Build CNN model with MobileNetV2
def build_cnn_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to draw bounding boxes around defect regions
def detect_defect_region(original_image, processed_image):
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_original = cv2.resize(gray_original, (224, 224))

    processed_gray = (processed_image[:, :, 0] * 255).astype(np.uint8)

    # Compute SSIM and get the difference map
    ssim_score, diff = ssim(gray_original, processed_gray, full=True, data_range=255)
    diff = (diff * 255).astype("uint8")  

    # Apply threshold to highlight defect regions
    _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around detected defect regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  

    return original_image, ssim_score

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

    # Callbacks for early stopping and saving the best model
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )
    
    model.save('ic_fault_detector_model.keras')
    print("Model training complete. Saved as 'ic_fault_detector_model.keras'.")

# Function to predict faults and highlight defect regions
def predict_fault():
    print("Please upload the images for fault detection:")
    uploaded_files = files.upload()
    image_paths = list(uploaded_files.keys())
    
    model = tf.keras.models.load_model('ic_fault_detector_model.keras')
    fault_types = ['Good', 'Scratch', 'Broken Legs']
    
    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        processed_image = preprocess_image(image_path)

        image_expanded = np.expand_dims(processed_image, axis=0)  
        prediction = model.predict(image_expanded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fault_type = fault_types[predicted_class]

        if fault_type != 'Good':  
            marked_image, ssim_score = detect_defect_region(original_image, processed_image)
            print(f"Fault Detected: {fault_type} (SSIM Score: {ssim_score:.2f})")
            cv2_imshow(marked_image)  
        else:
            print("No defects detected. The IC is in Good condition.")

# Main Workflow
upload_and_train()
predict_fault()
