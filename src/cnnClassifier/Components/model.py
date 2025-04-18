import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define paths
data_path = 'artifacts/data/'
model_save_path = 'artifacts/model/sign_language_model.h5'

# Image parameters
img_size = (64, 64)  # Resized image dimensions
batch_size = 32

# Data Augmentation and Image Preprocessing
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # 20% validation data
)

# Load training and validation datasets
train_generator = data_gen.flow_from_directory(
    data_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    data_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Mapping class indices to alphabet labels
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping for predictions
print("Class Labels:", class_labels)

num_classes = len(class_labels)  # Dynamically determine number of classes

# Define improved CNN model
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model with RMSprop optimizer
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[lr_scheduler]
)

# Save model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
