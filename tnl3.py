import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show shape
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# Show a sample image
plt.figure(figsize=(4,4))
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {class_names[y_train[0]]}")
plt.axis('off')
plt.show()

# Normalize pixel values to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN: (batch_size, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Split a validation set from training data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

print("Training set:", x_train.shape)
print("Validation set:", x_val.shape)
print("Test set:", x_test.shape)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val)
)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

predictions = model.predict(x_test)

def show_prediction(index):
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {class_names[np.argmax(predictions[index])]}, Actual: {class_names[y_test[index]]}")
    plt.axis('off')
    plt.show()

# Try a few predictions
for i in range(3):
    show_prediction(i)

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def prepare_image(img_path):
    img = Image.open(img_path)       # Load image (RGB)
    img = img.convert("L")            # Convert to grayscale
    img = ImageOps.invert(img)        # Invert colors for Fashion-MNIST style
    img = img.resize((28, 28))        # Resize to 28x28
    img_array = np.array(img) / 255.0 # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model
    return img_array

# Example:
img_path = "/content/trouser3.jpg"
img_prepared = prepare_image(img_path)

plt.imshow(img_prepared.reshape(28,28), cmap='gray')
plt.axis('off')
plt.title("Preprocessed Input Image")
plt.show()

pred = model.predict(img_prepared)
predicted_label = class_names[np.argmax(pred)]
print(f"Predicted class: {predicted_label}")
