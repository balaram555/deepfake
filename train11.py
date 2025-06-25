import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
train_dir = r'dataset/dataset/train'
val_dir = r'dataset/dataset/validation'
img_height, img_width = 128, 128
batch_size = 32
num_train_samples = 2000  # 1000 real + 1000 fake
num_val_samples = 400     # 200 real + 200 fake

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% for validation
)

# Create main generator for training data
main_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['real', 'fake'],
    shuffle=True,
    subset='training'  # Will be used for training
)

# Create validation generator from training directory
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    classes=['real', 'fake'],
    shuffle=False,
    subset='validation'  # Will be used for validation
)

# Model architecture
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

# Calculate steps based on sample counts
train_steps = num_train_samples // batch_size
val_steps = num_val_samples // batch_size

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

# Save the model in recommended Keras format
# Save in TensorFlow SavedModel format (folder with saved_model.pb)
model.save('deepfake_detector', include_optimizer=False)  # ← no file-ending!
print("✅ Model saved successfully in SavedModel format!")


# Evaluate on validation data
print("\nEvaluation results:")
val_results = model.evaluate(val_generator, steps=val_steps)
print(f"Validation Accuracy: {val_results[1]:.2%}")
print(f"Precision: {val_results[2]:.2%}")
print(f"Recall: {val_results[3]:.2%}")
