# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

import os
import json
import cv2

# --- 1. Configuration Setup ---
BASE_PATH = '/kaggle/input/datasets/pranay0/welding-dataset-coco' 
OUTPUT_BASE = '/kaggle/working/cnn_dataset'
folders_to_process = ['train', 'valid']

print("Starting Feature Localization (Cropping)...")

for folder in folders_to_process:
    DATA_DIR = os.path.join(BASE_PATH, folder)
    SAVE_DIR = os.path.join(OUTPUT_BASE, folder)

    json_file = next((os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.json')), None)
    
    if not json_file:
        print(f"⚠️ No JSON found in {folder}, skipping...")
        continue

    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'].replace(" ", "_") for cat in coco_data.get('categories', [])}
    images = {img['id']: img['file_name'] for img in coco_data.get('images', [])}

    for cat_name in categories.values():
        os.makedirs(os.path.join(SAVE_DIR, cat_name), exist_ok=True)

    annotations = coco_data.get('annotations', [])
    total_ann = len(annotations)
    print(f"\nProcessing {folder} set: Found {total_ann} annotations.")

    count = 0
    for i, ann in enumerate(annotations):
        # Progress update every 500 images so the console doesn't get too crowded
        if i % 500 == 0:
            print(f"Progress: {i}/{total_ann} crops completed...")

        img_name = images.get(ann['image_id'])
        if not img_name: continue

        img_path = os.path.join(DATA_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        x, y, w, h = map(int, ann['bbox'])
        crop = img[max(0, y):min(img.shape[0], y + h), max(0, x):min(img.shape[1], x + w)]
        
        if crop.size > 0:
            cat_name = categories.get(ann['category_id'], f'class_{ann["category_id"]}')
            save_path = os.path.join(SAVE_DIR, cat_name, f"crop_{i}.jpg")
            cv2.imwrite(save_path, crop)
            count += 1

    print(f"✅ Finished {folder}: Generated {count} crops.")

print("\n🚀 Stage 1 Complete!")

import tensorflow as tf

# --- 1. Define Parameters ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32 # If it crashes again, lower this to 16.

print("Loading Training Data from /kaggle/working...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/working/cnn_dataset/train',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nLoading Validation Data from /kaggle/working...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/working/cnn_dataset/valid',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"\n✅ Classes detected: {class_names}")

# --- 4. Pipeline Optimization (UPDATED) ---
AUTOTUNE = tf.data.AUTOTUNE

# REMOVED .cache() to save memory. Kept prefetch() to keep the GPU busy.
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("🚀 Stage 2 Complete: Memory-safe pipeline is ready.")

from tensorflow.keras import layers, models

# Number of classes from Stage 2
num_classes = len(class_names)

print("Rebuilding 4-Block Deep CNN Architecture...")

model = models.Sequential([
    # Input Layer (Matches the 150x150 size from Stage 2)
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255), 
    
    # Block 1: 32 Filters - Captures basic edges
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 2: 64 Filters - Captures basic shapes
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 3: 128 Filters - Captures defect textures
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 4: 256 Filters - Captures complex defect structures
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- Classification Head ---
    layers.Flatten(),
    
    # Fully connected layer (The "Brain")
    layers.Dense(256, activation='relu'), 
    
    # Regularization to prevent memory memorization
    layers.Dropout(0.5), 
    
    # Final Output Layer (Probabilities for each defect type)
    layers.Dense(num_classes, activation='softmax') 
])

# --- Compilation ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show the model architecture summary
model.summary()
print("✅ Stage 3 Complete: Architecture is built and ready for training!")

import matplotlib.pyplot as plt

# --- 1. Training Configuration ---
# 20 epochs is a great starting point for a dataset of this size.
EPOCHS = 20 

print(f"🚀 Starting Training for {EPOCHS} epochs...")
print("Note: Ensure your Kaggle GPU (T4 x2 or P100) is turned ON for this step!")

# --- 2. The Training Loop ---
# This is where the actual learning happens.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 3. Visualizing Results for the Project Report ---
# Extract the metrics from the training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] # Note: Keras usually calls this 'val_accuracy'
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2, color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2, color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2, color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, color='orange')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 4. Final Model Save ---
# This saves your trained weights so you don't have to train it again tomorrow!
model.save('welding_defect_cnn_model.h5')
print("\n✅ Training Complete! Model saved as 'welding_defect_cnn_model.h5'")



