import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-100 dataset with 'fine' labels
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


# Display 5 random samples from the training set
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    idx = np.random.randint(0, x_train.shape[0])
    axes[i].imshow(x_train[idx])
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# Scale images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#one-hot encode 
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# Build the CNN model
model = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#add earlystopping
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs= 30, batch_size=64, validation_data=(x_test, y_test), callbacks=[early_stopping])

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Select 5 random test images
indices = random.sample(range(len(x_test)), 5)
images = x_test[indices]
true_labels = np.argmax(y_test[indices], axis=1)
predictions = np.argmax(model.predict(images), axis=1)

# Class names for CIFAR-100
(_, label_names) = cifar100.load_data(label_mode='fine')

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    plt.title(f'True: {true_labels[i]}\nPred: {predictions[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()