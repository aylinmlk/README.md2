# 1. Gerekli kütüphaneleri ve modülleri içe aktar
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from   tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 2. Parametreleri tanımla
DATASET_DIR = "C:\\Users\\user\\PycharmProjects\\PythonProject6\\archive"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# 3. Veri artırma ve veri akışlarını oluştur
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print("Class indices:", train_generator.class_indices)

# 4. Model mimarisini tanımla
num_classes = 2  # Örneğin: maskeli ve maskesiz
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax') # Çoklu sınıflandırmaya uyumlu
])

# 5. Modeli derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Çoklu sınıflandırmaya uyumlu
              metrics=['accuracy'])

# 6. Modeli eğit
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=20)

# 7. Sonuçları görselleştir
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()