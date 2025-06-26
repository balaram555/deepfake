import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
train_dir = r'D:\DeepFake Project\dataset\train'
val_dir = r'D:\DeepFake Project\dataset\validation'
img_height, img_width = 128, 128
batch_size = 32

# Create data generators - REMOVED VALIDATION_SPLIT
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create training generator (uses ALL images)
main_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['real', 'fake'],
    shuffle=True
)

# Create separate validation generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,  # Use validation directory
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['real', 'fake'],
    shuffle=False
)

# Model architecture (unchanged)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='prec'), 
             tf.keras.metrics.Recall(name='rec')]
)

# Calculate steps based on ACTUAL image counts
train_steps = main_generator.samples // batch_size
val_steps = val_generator.samples // batch_size

# Train the model
history = model.fit(
    main_generator,
    steps_per_epoch=train_steps,
    epochs=30,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]
)

# Save the model
model.save('deepfake_detector.keras')
print(f"Model trained on {main_generator.samples} images")
print(f"Validated on {val_generator.samples} images")

# Evaluate on validation data
print("\nEvaluation results:")
val_results = model.evaluate(val_generator, steps=val_steps)
print(f"Validation Accuracy: {val_results[1]:.2%}")
print(f"Precision: {val_results[2]:.2%}")
print(f"Recall: {val_results[3]:.2%}")
