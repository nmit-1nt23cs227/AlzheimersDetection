import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocessing import load_data
from model import build_cnn

# Ensure model directory exists
os.makedirs('saved_model', exist_ok=True)

# Load data
train_gen, val_gen, test_gen = load_data()

# Build model
model = build_cnn(input_shape=(224, 224, 3), num_classes=4)

# Callbacks
checkpoint_path = 'saved_model/best_model.h5'  # ✔ inside project folder
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")