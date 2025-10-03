import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Define image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 10 
NUM_CLASSES = 2 # 0: Benign, 1: Malignant
2. Prepare Data (Using ImageDataGenerator)
Assuming you have a directory structure like this for your dataset (e.g., from a source like Kaggle's Skin Cancer datasets):

base_dir/
    train/
        benign/
            img1.jpg
            img2.jpg
            ...
        malignant/
            imgA.jpg
            imgB.jpg
            ...
    validation/
        benign/
            img3.jpg
            ...
        malignant/
            imgC.jpg
            ...
Python

# --- REPLACE 'path/to/your/data' with your actual dataset directory ---
data_dir = 'path/to/your/data' 
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
# -------------------------------------------------------------------

# Data Augmentation and Normalization for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values to [0, 1]
    rotation_range=20,          # Data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only Normalization for Validation (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary' # For binary classification (0 or 1)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"Class indices: {train_generator.class_indices}") 
# Expected: {'benign': 0, 'malignant': 1} (or similar)
3. Build the CNN Model
A simple Sequential CNN model for image classification. More complex models, like transfer learning with EfficientNet or ResNet, are often preferred for medical image analysis.

Python

def create_cnn_model(input_shape):
    model = Sequential([
        # First Conv Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second Conv Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Conv Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten layer to transition to Dense layers
        Flatten(),
        
        # Fully Connected (Dense) layers
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(512, activation='relu'),
        
        # Output layer (1 neuron for binary classification)
        Dense(1, activation='sigmoid') # Sigmoid for binary output (0 or 1)
    ])
    return model

input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
model = create_cnn_model(input_shape)
4. Compile and Train the Model
Python

# Compile the model
model.compile(
    optimizer='adam',
    # Use binary_crossentropy for binary classification
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC is a great metric for imbalanced data
)

# Display the model architecture
model.summary()

# Train the model
# NOTE: This step requires a lot of time and a suitable GPU/CPU environment.
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE
# )

print("\nModel setup complete. Uncomment the 'model.fit' block to start training.")
5. Prediction (Example)
After training and saving your model (e.g., model.save('skin_cancer_detector.h5')), you can use it to make a prediction on a new image.

Python

# Load the trained model (assuming it's saved)
# model = tf.keras.models.load_model('skin_cancer_detector.h5')

def predict_skin_lesion(image_path, model, img_height, img_width):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand dimensions to match model input shape (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0) 
    # Normalize
    img_array /= 255.0 

    # Make the prediction
    prediction = model.predict(img_array)
    probability_malignant = prediction[0][0]
    
    # Classify based on a threshold (e.g., 0.5)
    if probability_malignant > 0.5:
        result = "Malignant"
    else:
        result = "Benign"
        
    return result, probability_malignant

# Example usage (uncomment after model is trained/loaded)
# sample_image_path = 'path/to/new/lesion_image.jpg'
# predicted_class, probability = predict_skin_lesion(model, sample_image_path, IMG_HEIGHT, IMG_WIDTH)
# print(f"Image {sample_image_path} is predicted as: {predicted_class}")
# print(f"Probability of Malignancy: {probability:.4f}")
