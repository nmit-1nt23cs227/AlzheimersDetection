import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODEL_PATH = '../saved_model/best_model.h5'
TEST_DIR = '../data_split/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# Data generator for test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# Classification report
print("\n📊 Classification Report:")
from sklearn.utils.multiclass import unique_labels

present_labels = sorted(list(unique_labels(y_true, y_pred)))
present_class_names = [CLASS_NAMES[i] for i in present_labels]

print(classification_report(y_true, y_pred, target_names=present_class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("🧠 Confusion Matrix: Alzheimer's Detection")
plt.tight_layout()
plt.show()
